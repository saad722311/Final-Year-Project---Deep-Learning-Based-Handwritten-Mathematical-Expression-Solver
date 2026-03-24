from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.end2end.latex_tokenizer import build_vocab
from src.end2end.dataset import CrohmeE2EDataset
from src.end2end.model import E2ETransformer


def get_device(d: str) -> torch.device:
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(batch, pad_id: int):
    xs, ys, paths = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,1,H,W)
    maxlen = max(y.size(0) for y in ys)
    ypad = torch.full((len(ys), maxlen), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        ypad[i, : y.size(0)] = y
    return x, ypad, list(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_hw", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--max_vocab", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print("Device:", device)

    # build vocab from train truth latex
    train_latex = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            # assume inkml path stored, truth pulled from file later, but we can just build from inkml truth at runtime
            # Keep it simple: build later from dataset reading truth.
            # Here we just store paths.
            pass

    # easiest: build vocab by scanning inkml files listed in train jsonl
    inkml_paths = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            p = o.get("inkml_path") or o.get("inkml") or o.get("path")
            if p:
                inkml_paths.append(p)

    from src.end2end.render_inkml_to_image import read_truth_latex
    for p in inkml_paths:
        train_latex.append(read_truth_latex(p))

    vocab = build_vocab(train_latex, max_vocab=args.max_vocab)
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, indent=2)

    train_ds = CrohmeE2EDataset(args.train_jsonl, vocab=vocab, image_hw=tuple(args.image_hw))
    val_ds = CrohmeE2EDataset(args.val_jsonl, vocab=vocab, image_hw=tuple(args.image_hw))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate(b, vocab.pad))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, vocab.pad))

    model = E2ETransformer(vocab_size=len(vocab.itos), pad_id=vocab.pad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e18

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y, _ in train_dl:
            x = x.to(device)
            y = y.to(device)

            # teacher forcing: input is y[:, :-1], target is y[:, 1:]
            y_in = y[:, :-1]
            y_tg = y[:, 1:]

            logits = model(x, y_in)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_tg.reshape(-1),
                ignore_index=vocab.pad,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())

        tr_loss /= max(1, len(train_dl))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device)
                y = y.to(device)
                y_in = y[:, :-1]
                y_tg = y[:, 1:]
                logits = model(x, y_in)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_tg.reshape(-1),
                    ignore_index=vocab.pad,
                )
                va_loss += float(loss.item())
        va_loss /= max(1, len(val_dl))

        print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            ckpt = {
                "model": model.state_dict(),
                "itos": vocab.itos,
                "pad": vocab.pad,
                "bos": vocab.bos,
                "eos": vocab.eos,
                "unk": vocab.unk,
                "image_hw": args.image_hw,
            }
            torch.save(ckpt, out / "best.pt")
            print("  saved best ->", out / "best.pt")


if __name__ == "__main__":
    main()