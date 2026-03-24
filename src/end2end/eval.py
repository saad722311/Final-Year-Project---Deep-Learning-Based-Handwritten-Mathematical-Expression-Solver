# src/end2end/eval.py
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from src.end2end.dataset import CrohmeE2EDataset
from src.end2end.model import E2ETransformer
from src.end2end.render_inkml_to_image import read_truth_latex


# -------------------------
# Device
# -------------------------
def get_device(d: str) -> torch.device:
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Minimal "vocab" wrapper (from ckpt)
# -------------------------
class Vocab:
    def __init__(self, itos: List[str], pad: int, bos: int, eos: int, unk: int):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad = int(pad)
        self.bos = int(bos)
        self.eos = int(eos)
        self.unk = int(unk)

    def encode(self, toks: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = [int(self.stoi.get(t, self.unk)) for t in toks]
        if add_bos_eos:
            return [self.bos] + ids + [self.eos]
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            i = int(i)
            if i in (self.pad, self.bos):
                continue
            if i == self.eos:
                break
            if 0 <= i < len(self.itos):
                toks.append(self.itos[i])
            else:
                toks.append(self.itos[self.unk] if 0 <= self.unk < len(self.itos) else "<UNK>")
        return detokenize_latex(toks)


# -------------------------
# Token -> LaTeX string
# -------------------------
def detokenize_latex(tokens: List[str]) -> str:
    # CROHME latex is usually space-free; keep it that way
    return "".join(tokens) if tokens else ""


def normalize_latex(s: str) -> str:
    s = (s or "").strip()

    # remove surrounding $...$ or $$...$$
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # remove \\left / \\right
    s = s.replace("\\left", "").replace("\\right", "")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_latex_strict(s: str) -> str:
    s = normalize_latex(s)
    return s.replace(" ", "")


# -------------------------
# Collate
# -------------------------
def collate(batch, pad_id: int):
    xs, ys, paths = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,1,H,W)
    maxlen = max(y.size(0) for y in ys)
    ypad = torch.full((len(ys), maxlen), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        ypad[i, : y.size(0)] = y
    return x, ypad, list(paths)


# -------------------------
# Greedy decoding
# -------------------------
@torch.no_grad()
def greedy_decode(
    model: E2ETransformer,
    x: torch.Tensor,
    vocab: Vocab,
    max_len: int,
    device: torch.device,
) -> List[int]:
    """
    x: (1,1,H,W)
    returns list of token ids including BOS ... EOS
    """
    model.eval()
    x = x.to(device)

    ys = torch.tensor([[vocab.bos]], dtype=torch.long, device=device)  # (1, t)

    for _ in range(max_len - 1):
        logits = model(x, ys)  # (1, t, vocab_size)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == vocab.eos:
            break

    return ys.squeeze(0).tolist()


# -------------------------
# Beam search decoding
# -------------------------
@torch.no_grad()
def beam_search_decode(
    model: E2ETransformer,
    x: torch.Tensor,           # (1,1,H,W)
    vocab: Vocab,
    max_len: int,
    device: torch.device,
    beam_size: int = 5,
) -> List[int]:
    """
    Returns best token id sequence including BOS ... EOS.
    """
    model.eval()
    x = x.to(device)

    # each beam: (tokens, logprob)
    beams: List[Tuple[List[int], float]] = [([vocab.bos], 0.0)]

    for _step in range(max_len - 1):
        new_beams: List[Tuple[List[int], float]] = []

        for tokens, score in beams:
            # already ended
            if tokens[-1] == vocab.eos:
                new_beams.append((tokens, score))
                continue

            y = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
            logits = model(x, y)  # (1,T,V)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # (V,)

            topk = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_tok = int(topk.indices[i].item())
                next_score = score + float(topk.values[i].item())
                new_beams.append((tokens + [next_tok], next_score))

        # keep top beam_size
        new_beams.sort(key=lambda t: t[1], reverse=True)
        beams = new_beams[:beam_size]

        # stop if all ended
        if all(toks[-1] == vocab.eos for toks, _ in beams):
            break

    return beams[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--in_jsonl", required=True, help="val jsonl (same format as train)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=180, help="max decoded tokens")
    ap.add_argument("--beam_size", type=int, default=5, help="0=greedy, >=2=beam search")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = Vocab(
        itos=ckpt["itos"],
        pad=ckpt["pad"],
        bos=ckpt["bos"],
        eos=ckpt["eos"],
        unk=ckpt["unk"],
    )
    image_hw = tuple(int(x) for x in ckpt.get("image_hw", [256, 256]))

    ds = CrohmeE2EDataset(args.in_jsonl, vocab=vocab, image_hw=image_hw)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, vocab.pad),
    )

    model = E2ETransformer(vocab_size=len(vocab.itos), pad_id=vocab.pad)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rows = []
    n = 0
    em = 0
    nem = 0

    for x, _, paths in dl:
        for i in range(x.size(0)):
            if args.limit and n >= args.limit:
                break

            xi = x[i : i + 1]  # (1,1,H,W)
            inkml_path = paths[i]

            if args.beam_size and args.beam_size >= 2:
                pred_ids = beam_search_decode(
                    model,
                    xi,
                    vocab=vocab,
                    max_len=args.max_len,
                    device=device,
                    beam_size=args.beam_size,
                )
            else:
                pred_ids = greedy_decode(
                    model,
                    xi,
                    vocab=vocab,
                    max_len=args.max_len,
                    device=device,
                )

            pred = vocab.decode_ids(pred_ids)
            gt = read_truth_latex(inkml_path)

            pred_norm = normalize_latex(pred)
            gt_norm = normalize_latex(gt)

            pred_n = normalize_latex_strict(pred)
            gt_n = normalize_latex_strict(gt)

            is_em = int(pred_norm == gt_norm)
            is_nem = int(pred_n == gt_n)

            em += is_em
            nem += is_nem
            n += 1

            rows.append(
                {
                    "inkml_path": inkml_path,
                    "gt": gt_norm,
                    "pred": pred_norm,
                    "EM": is_em,
                    "nEM": is_nem,
                }
            )

        if args.limit and n >= args.limit:
            break

    em_score = em / max(1, n)
    nem_score = nem / max(1, n)

    out_csv = out_dir / "predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["inkml_path", "gt", "pred", "EM", "nEM"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_txt = out_dir / "eval.txt"
    out_txt.write_text(
        f"items={n}\nEM={em_score:.4f}\nnEM={nem_score:.4f}\n",
        encoding="utf-8",
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(out_txt.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()