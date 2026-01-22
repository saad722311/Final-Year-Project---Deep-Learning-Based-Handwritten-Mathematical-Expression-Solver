# src/train/infer.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.tokenizer import CharTokenizer
from src.data.datasets import CROHMEProcessedConfig, CROHMEProcessedDataset
from src.data.transforms import ImageTransformConfig
from src.data.collate import HMERBatchCollator
from src.models.hmer_model import HMERModel


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.startswith(("encoder.", "decoder.")) for k in ckpt.keys()):
        state = ckpt
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    model.load_state_dict(state, strict=True)


# -----------------------
# Metrics helpers
# -----------------------
_WS = re.compile(r"\s+")


def normalize_latex(s: str) -> str:
    """Light canonicalization (avoid EM=0 due to spaces/$)."""
    s = s.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = _WS.sub("", s)
    return s


def count_exact(gt: list[str], pred: list[str]) -> int:
    return sum(1 for g, p in zip(gt, pred) if g.strip() == p.strip())


def count_norm_exact(gt: list[str], pred: list[str]) -> int:
    return sum(1 for g, p in zip(gt, pred) if normalize_latex(g) == normalize_latex(p))


def token_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> tuple[int, int]:
    """
    logits: (B,L,V)
    targets: (B,L)
    Returns (correct_tokens, total_tokens) ignoring PAD.
    """
    pred = logits.argmax(dim=-1)  # (B,L)
    mask = (targets != pad_id)
    correct = ((pred == targets) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default=None)

    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--out", type=str, default=None)

    ap.add_argument("--eval_all", action="store_true")
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=50)

    # decoding controls
    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.10)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--forbid_unk", action="store_true", help="Forbid UNK during decoding")

    ap.add_argument("--no_teacher_metrics", action="store_true", help="Skip teacher-forcing token acc (debug only)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    device = get_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    run_name = cfg["run"]["name"]
    output_dir = Path(cfg["run"]["output_dir"]) / run_name

    tok_path = output_dir / "tokenizer.json"
    tokenizer = CharTokenizer.load(tok_path)
    print(f"Loaded tokenizer. Vocab size: {tokenizer.vocab_size}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    split = args.split or cfg["data"].get("valid_split", "valid")

    tf_cfg = ImageTransformConfig(
        target_height=int(cfg["data"]["target_height"]),
        max_width=int(cfg["data"]["max_width"]),
        invert=bool(cfg["data"].get("invert", False)),
    )

    ds = CROHMEProcessedDataset(
        cfg=CROHMEProcessedConfig(processed_dir=str(processed_dir), split=split),
        tokenizer=tokenizer,
        img_tf_cfg=tf_cfg,
        debug_print=False,
    )

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])
    collator = HMERBatchCollator(pad_id=tokenizer.pad_id)

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=not args.eval_all,
        num_workers=nw,
        collate_fn=collator,
        persistent_workers=(nw > 0),
    )

    model = HMERModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
        unk_id=tokenizer.unk_id,

        encoder_d_model=int(cfg["model"]["d_model"]),
        decoder_hidden=int(cfg["model"].get("hidden_size", 256)),

        # ✅ NEW: supports transformer too
        decoder_type=str(cfg["model"].get("decoder_type", "lstm")),
        n_heads=int(cfg["model"].get("n_heads", 4)),
        n_layers=int(cfg["model"].get("n_layers", 4)),
        ff_dim=int(cfg["model"].get("ff_dim", 1024)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        max_len=int(cfg["data"].get("max_decode_len", 256)),
    ).to(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else (output_dir / "best.pt")
    load_checkpoint(model, ckpt_path, device=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()

    lines: list[str] = []
    printed = 0

    total_samples = 0
    em_correct = 0
    nem_correct = 0

    tok_correct = 0
    tok_total = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        image_widths = batch.get("image_widths", None)
        if image_widths is not None:
            image_widths = image_widths.to(device)

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        gts = batch["labels"]
        filenames = batch["filenames"]

        # Teacher-forcing token accuracy
        if not args.no_teacher_metrics:
            logits = model(images=images, input_ids=input_ids, image_widths=image_widths)
            c, t = token_accuracy_from_logits(logits, target_ids, pad_id=tokenizer.pad_id)
            tok_correct += c
            tok_total += t

        # Decode
        pred_ids = model.generate(
            images,
            image_widths=image_widths,
            max_len=args.max_len,
            decode=args.decode,
            beam_size=args.beam_size,
            alpha=args.alpha,
            min_len=args.min_len,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            forbid_unk=bool(args.forbid_unk),
        )
        pred_ids_list = pred_ids.detach().cpu().tolist()
        preds = [tokenizer.decode(ids, remove_special=True, stop_at_eos=True) for ids in pred_ids_list]

        if args.eval_all:
            em_correct += count_exact(gts, preds)
            nem_correct += count_norm_exact(gts, preds)
            total_samples += len(gts)

            if args.max_eval is not None and total_samples >= args.max_eval:
                break

            if (batch_idx + 1) % args.print_every == 0:
                em = 100.0 * em_correct / max(1, total_samples)
                nem = 100.0 * nem_correct / max(1, total_samples)
                ta = 100.0 * tok_correct / max(1, tok_total) if tok_total > 0 else 0.0
                print(f"[eval] batches={batch_idx+1} samples={total_samples} EM={em:.2f}% nEM={nem:.2f}% tokAcc={ta:.2f}%")

        # Print samples
        for i in range(len(gts)):
            if printed >= args.num_samples:
                break
            msg = (
                f"---\n"
                f"file: {filenames[i]}\n"
                f"GT  : {gts[i]}\n"
                f"PRED: {preds[i]}\n"
            )
            print(msg)
            lines.append(msg)
            printed += 1

        if not args.eval_all and printed >= args.num_samples:
            break

    if args.eval_all:
        em = 100.0 * em_correct / max(1, total_samples)
        nem = 100.0 * nem_correct / max(1, total_samples)
        ta = 100.0 * tok_correct / max(1, tok_total) if tok_total > 0 else 0.0
        limit_note = f" (first {total_samples})" if args.max_eval is not None else ""
        print(f"\nExact-match ({split}){limit_note}: {em:.2f}% ({em_correct}/{total_samples})")
        print(f"Norm exact-match ({split}){limit_note}: {nem:.2f}% ({nem_correct}/{total_samples})")
        if not args.no_teacher_metrics:
            print(f"Token accuracy ({split}){limit_note}: {ta:.2f}% ({tok_correct}/{tok_total})")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved outputs to: {out_path}")


if __name__ == "__main__":
    main()