# src/symbol_recognition/eval.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import tempfile

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.symbol_recognition.dataset import SymbolDataset, SymbolDatasetConfig
from src.symbol_recognition.model import TinySymbolCNN


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    # NEW: what to do with unseen labels
    ap.add_argument(
        "--unseen",
        type=str,
        default="skip",
        choices=["skip", "unk"],
        help="How to handle labels not present in checkpoint vocab. "
             "'skip' drops those rows. 'unk' maps them to <UNK> if present.",
    )
    # NEW: optional save of filtered CSV + unseen report
    ap.add_argument("--filtered_csv_out", type=str, default="")
    ap.add_argument("--unseen_out", type=str, default="")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stoi: dict[str, int] = ckpt["stoi"]
    itos: list[str] = ckpt["itos"]
    image_size = int(ckpt.get("image_size", 64))

    # -----------------------------
    # Filter / handle unseen labels
    # -----------------------------
    df = pd.read_csv(args.labels_csv)
    df["label"] = df["label"].astype(str)

    known = set(stoi.keys())
    mask_known = df["label"].isin(known)

    dropped = int((~mask_known).sum())
    kept = int(mask_known.sum())

    unseen_labels = sorted(set(df.loc[~mask_known, "label"].tolist()))

    if dropped > 0:
        if args.unseen == "skip":
            print(f"[eval] unseen labels detected -> skipping them. kept={kept} dropped={dropped}")
        else:
            # map to unk if possible
            unk_candidates = ["<unk>", "<UNK>", "[UNK]", "UNK"]
            unk_token = next((t for t in unk_candidates if t in stoi), None)
            if unk_token is None:
                print("[eval] --unseen=unk requested, but checkpoint has no UNK token. Falling back to skip.")
                args.unseen = "skip"
                print(f"[eval] skipping unseen. kept={kept} dropped={dropped}")
            else:
                print(f"[eval] mapping unseen labels -> {unk_token}. total_unseen_rows={dropped}")
                df.loc[~mask_known, "label"] = unk_token
                # no dropping now
                kept = len(df)
                dropped = 0
                unseen_labels = []  # handled by mapping

    # save unseen list if requested
    if args.unseen_out:
        outp = Path(args.unseen_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(unseen_labels) + ("\n" if unseen_labels else ""), encoding="utf-8")
        print(f"[eval] unseen labels list saved to: {outp}")

    # save filtered csv if requested
    if args.filtered_csv_out:
        outp = Path(args.filtered_csv_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if args.unseen == "skip":
            df_use = df[mask_known].copy()
        else:
            df_use = df.copy()
        df_use.to_csv(outp, index=False)
        print(f"[eval] filtered labels csv saved to: {outp}")

    # create a temp csv for the dataset to read
    if args.unseen == "skip":
        df = df[mask_known].copy()

    if len(df) == 0:
        raise SystemExit("[eval] No samples left after filtering. Cannot evaluate.")

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8") as tmp:
        df.to_csv(tmp.name, index=False)
        labels_csv_for_ds = tmp.name

    # -------------------
    # Dataset + Loader
    # -------------------
    ds = SymbolDataset(
        SymbolDatasetConfig(
            images_dir=args.images_dir,
            labels_csv=labels_csv_for_ds,
            image_size=image_size,
            invert=False,
        )
    )

    # force checkpoint label map
    ds.stoi = stoi
    ds.itos = itos
    ds.num_classes = len(stoi)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -------------------
    # Model
    # -------------------
    model = TinySymbolCNN(num_classes=len(stoi))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    conf = defaultdict(int)

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        total += int(x.size(0))

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

        for g, p in zip(y.tolist(), pred.tolist()):
            if g != p:
                conf[(g, p)] += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top = sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:50]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt", "pred", "count"])
        for (g, p), c in top:
            w.writerow([itos[g], itos[p], c])

    print(f"Eval: loss={avg_loss:.4f} acc={acc:.4f} (n={total})")
    if dropped > 0 and args.unseen == "skip":
        print(f"[eval] skipped unseen rows: {dropped}")
        if unseen_labels:
            print(f"[eval] unseen labels: {unseen_labels}")
    print(f"Top confusions saved to: {out_path}")


if __name__ == "__main__":
    main()