from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
    return None


# Accepts: form_045_E353__sym000  (NO extension)
_SYM_RE = re.compile(r"^(?P<ui>.+?)__sym_?(?P<idx>\d+)$", flags=re.IGNORECASE)


def _stem_no_ext(x: str) -> str:
    """
    Handles inputs like:
      form_045_E353__sym000.png
      /some/path/form_045_E353__sym000.png
      form_045_E353__sym000
    """
    x = (x or "").strip()
    # take last path component if any, then remove extension
    return Path(x).name.rsplit(".", 1)[0]  # safer than .stem for weird multi-dots


def _split_name_to_ui_and_idx(name: str) -> tuple[str, int]:
    """
    Preferred: use the `name` column like form_045_E353__sym000(.png)
    Returns (base_ui, sym_idx).
    """
    stem = _stem_no_ext(name)
    m = _SYM_RE.match(stem)
    if m:
        return m.group("ui"), int(m.group("idx"))
    return stem, 0


def _split_filename_to_ui_and_idx(fn: str) -> tuple[str, int]:
    stem = _stem_no_ext(fn)
    m = re.search(r"__sym_?(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        return stem[: m.start()], int(m.group(1))
    m = re.search(r"[_-](\d+)$", stem)
    if m:
        return stem[: m.start()], int(m.group(1))
    return stem, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Full predictions CSV (the *_predictions.csv file)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all expressions")
    ap.add_argument("--sep", default=" ", help="Separator between tokens in output string")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)

    fn_col = _pick_col(df.columns.tolist(), ["name", "filename", "file", "image", "img", "path"])
    pred_col = _pick_col(df.columns.tolist(), ["y_pred", "pred", "pred_label"])
    gt_col = _pick_col(df.columns.tolist(), ["y_true", "gt", "label", "truth"])

    if fn_col is None or pred_col is None:
        raise SystemExit(
            f"Could not detect required columns.\n"
            f"Found columns: {list(df.columns)}\n"
            f"Need at least: name/filename + pred"
        )

    if fn_col.lower() == "name":
        base_idx = df[fn_col].astype(str).apply(_split_name_to_ui_and_idx)
    else:
        base_idx = df[fn_col].astype(str).apply(_split_filename_to_ui_and_idx)

    df["_ui"] = [u for (u, _) in base_idx]
    df["_idx"] = [i for (_, i) in base_idx]

    rows_out = []
    for ui, g in df.groupby("_ui", sort=False):
        g2 = g.sort_values("_idx", kind="stable")

        pred_tokens = [str(x) for x in g2[pred_col].tolist()]
        pred_join = args.sep.join(pred_tokens)

        gt_join = ""
        em = ""
        if gt_col is not None:
            gt_tokens = [str(x) for x in g2[gt_col].tolist()]
            gt_join = args.sep.join(gt_tokens)
            em = int(pred_join == gt_join)

        rows_out.append(
            {
                "ui": ui,
                "n_symbols": len(g2),
                "pred_linear": pred_join,
                "gt_linear": gt_join,
                "em": em,
            }
        )

    out_df = pd.DataFrame(rows_out)
    if args.limit and args.limit > 0:
        out_df = out_df.head(args.limit)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[linear] pred_csv: {args.pred_csv}")
    print(f"[linear] expressions: {len(out_df)}")
    if "em" in out_df.columns and out_df["em"].astype(str).str.len().sum() > 0:
        try:
            em_rate = float(out_df["em"].mean())
            print(f"[linear] EM (linear tokens): {em_rate:.4f}")
        except Exception:
            pass
    print(f"[linear] saved: {out_path}")


if __name__ == "__main__":
    main()