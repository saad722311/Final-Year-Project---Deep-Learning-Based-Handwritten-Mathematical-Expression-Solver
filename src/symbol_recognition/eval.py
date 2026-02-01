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
import torch.nn.functional as F
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


def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    # Backwards-compatible: this stays as "top confusions csv"
    ap.add_argument("--out", type=str, required=True, help="Output CSV for TOP confusions (gt,pred,count).")

    # NEW: full predictions output
    ap.add_argument(
        "--out_predictions",
        type=str,
        default="",
        help="Output CSV for FULL per-sample predictions. "
             "If empty, defaults to <out_stem>_predictions.csv in same folder.",
    )

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    # What to do with unseen labels
    ap.add_argument(
        "--unseen",
        type=str,
        default="skip",
        choices=["skip", "unk"],
        help="How to handle labels not present in checkpoint vocab. "
             "'skip' drops those rows. 'unk' maps them to <UNK> if present.",
    )
    ap.add_argument("--filtered_csv_out", type=str, default="")
    ap.add_argument("--unseen_out", type=str, default="")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # -------------------
    # Load checkpoint
    # -------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    stoi: dict[str, int] = ckpt["stoi"]
    itos: list[str] = ckpt["itos"]
    image_size = int(ckpt.get("image_size", 64))

    # -----------------------------
    # Filter / handle unseen labels
    # -----------------------------
    df = pd.read_csv(args.labels_csv)
    if "label" not in df.columns:
        raise SystemExit("[eval] labels_csv must contain a 'label' column.")
    df["label"] = df["label"].astype(str)

    known = set(stoi.keys())
    mask_known = df["label"].isin(known)

    dropped = int((~mask_known).sum())
    kept = int(mask_known.sum())
    unseen_labels = sorted(set(df.loc[~mask_known, "label"].tolist()))

    if dropped > 0:
        if args.unseen == "skip":
            print(f"[eval] unseen labels detected -> skipping. kept={kept} dropped={dropped}")
        else:
            # map to UNK if possible
            unk_candidates = ["<unk>", "<UNK>", "[UNK]", "UNK"]
            unk_token = next((t for t in unk_candidates if t in stoi), None)
            if unk_token is None:
                print("[eval] --unseen=unk requested, but checkpoint has no UNK token. Falling back to skip.")
                args.unseen = "skip"
                print(f"[eval] skipping unseen. kept={kept} dropped={dropped}")
            else:
                print(f"[eval] mapping unseen labels -> {unk_token}. total_unseen_rows={dropped}")
                df.loc[~mask_known, "label"] = unk_token
                kept = len(df)
                dropped = 0
                unseen_labels = []

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

    # apply skip filtering
    if args.unseen == "skip":
        df = df[mask_known].copy()

    if len(df) == 0:
        raise SystemExit("[eval] No samples left after filtering. Cannot evaluate.")

    # temp csv for dataset
    tmp_path = None
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    # -------------------
    # Dataset + Loader
    # -------------------
    ds = SymbolDataset(
        SymbolDatasetConfig(
            images_dir=args.images_dir,
            labels_csv=tmp_path,
            image_size=image_size,
            invert=False,
        )
    )

    # Force checkpoint vocab
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

    # confusion counts in ID space
    conf = defaultdict(int)

    # full predictions rows
    pred_rows = []

    # -------------------
    # Eval loop
    # -------------------
    for batch in loader:
        # Your dataset returns: x, y, something, something
        # We'll handle defensively.
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            meta1 = batch[2] if len(batch) > 2 else None  # often filename/path
            meta2 = batch[3] if len(batch) > 3 else None
        else:
            raise SystemExit("[eval] Unexpected batch format from DataLoader.")

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        bs = int(x.size(0))
        total_loss += float(loss.item()) * bs
        total += bs

        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        correct_mask = (pred == y)
        correct += int(correct_mask.sum().item())

        # gather per-sample confidence of predicted class
        pred_conf = probs.gather(1, pred.view(-1, 1)).squeeze(1)  # (B,)

        # update confusion pairs
        for g, p in zip(y.tolist(), pred.tolist()):
            if g != p:
                conf[(g, p)] += 1

        # try to extract filenames/ids
        # meta1 is often a list of paths/strings
        names = None
        if meta1 is not None:
            # If it's already list-like with same batch size
            if isinstance(meta1, (list, tuple)) and len(meta1) == bs:
                names = [_safe_str(v) for v in meta1]
            else:
                # Sometimes it's a tensor of indices etc.
                try:
                    if hasattr(meta1, "tolist"):
                        m = meta1.tolist()
                        if isinstance(m, list) and len(m) == bs:
                            names = [_safe_str(v) for v in m]
                except Exception:
                    pass

        if names is None:
            names = ["" for _ in range(bs)]

        # save full predictions rows
        for i in range(bs):
            gt_id = int(y[i].item())
            pr_id = int(pred[i].item())
            pred_rows.append({
                "name": names[i],
                "y_true": itos[gt_id],
                "y_pred": itos[pr_id],
                "correct": int(correct_mask[i].item()),
                "conf": float(pred_conf[i].item()),
            })

    # Cleanup temp csv
    try:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # -------------------
    # Outputs
    # -------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Top confusions
    top = sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:50]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt", "pred", "count"])
        for (g, p), c in top:
            w.writerow([itos[g], itos[p], c])

    # 2) Full predictions csv
    if args.out_predictions.strip():
        pred_path = Path(args.out_predictions)
    else:
        pred_path = out_path.with_name(out_path.stem + "_predictions.csv")

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    print(f"Eval: loss={avg_loss:.4f} acc={acc:.4f} (n={total})")
    if dropped > 0 and args.unseen == "skip":
        print(f"[eval] skipped unseen rows: {dropped}")
        if unseen_labels:
            print(f"[eval] unseen labels: {unseen_labels}")
    print(f"[eval] Top confusions saved to: {out_path}")
    print(f"[eval] Full predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()