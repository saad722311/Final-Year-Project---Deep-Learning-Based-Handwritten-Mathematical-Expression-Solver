from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def clip(s: object, n: int = 70) -> str:
    s = "" if pd.isna(s) else str(s)
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear_csv", required=True)
    ap.add_argument("--rule_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_wins", type=int, default=2)
    ap.add_argument("--n_improve", type=int, default=3)
    ap.add_argument("--n_fail", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load data ---
    lin = pd.read_csv(args.linear_csv)
    rb = pd.read_csv(args.rule_csv)

    # --- sanity checks ---
    required_lin = {"ui", "pred_linear", "em"}
    if not required_lin.issubset(lin.columns):
        raise SystemExit(f"[baselineB] linear_csv missing {required_lin - set(lin.columns)}")

    required_rb = {"pred", "gt", "em", "inkml_path"}
    if not required_rb.issubset(rb.columns):
        raise SystemExit(f"[baselineB] rule_csv missing {required_rb - set(rb.columns)}")

    # rename for clarity
    lin = lin.rename(columns={"em": "em_lin"})
    rb = rb.rename(columns={"em": "em_rb", "pred": "pred_rb", "gt": "gt_rb"})

    # extract ui from inkml filename
    rb["ui"] = rb["inkml_path"].astype(str).apply(lambda x: Path(x).stem)

    # --- merge ---
    m = lin.merge(
        rb[["ui", "pred_rb", "gt_rb", "em_rb", "inkml_path"]],
        on="ui",
        how="inner",
    )

    print("merged_rows:", len(m))
    if len(m) == 0:
        raise SystemExit("[baselineB] merge produced 0 rows — check ui alignment")

    m["em_lin"] = m["em_lin"].astype(int)
    m["em_rb"] = m["em_rb"].astype(int)

    # --- pick examples ---
    wins = m[(m["em_lin"] == 1) & (m["em_rb"] == 1)]
    improve = m[(m["em_lin"] == 0) & (m["em_rb"] == 1)]
    fails = m[(m["em_rb"] == 0)]

    take = []
    take += wins.head(args.n_wins)["ui"].tolist()
    take += improve.head(args.n_improve)["ui"].tolist()
    take += fails.head(args.n_fail)["ui"].tolist()

    need = args.n_wins + args.n_improve + args.n_fail
    if len(take) < need:
        rest = m[~m["ui"].isin(take)].head(need - len(take))["ui"].tolist()
        take += rest

    take = take[:need]

    # save picked UI list
    pick_path = out_dir / "picked_ui.txt"
    pick_path.write_text("\n".join(take), encoding="utf-8")
    print("saved:", pick_path)

    print("\npicked examples:")
    for u in take:
        r = m[m["ui"] == u].iloc[0]
        print(f"- {u} | lin_em={r['em_lin']} rb_em={r['em_rb']}")

    # --- build table ---
    show = m[m["ui"].isin(take)].copy()
    show["order"] = show["ui"].apply(lambda u: take.index(u))
    show = show.sort_values("order")

    table_df = pd.DataFrame({
        "UI": show["ui"].tolist(),
        "GT": [clip(x) for x in show["gt_rb"].tolist()],
        "Linear": [clip(x) for x in show["pred_linear"].tolist()],
        "Rule-based": [clip(x) for x in show["pred_rb"].tolist()],
        "EM_lin": show["em_lin"].tolist(),
        "EM_rb": show["em_rb"].tolist(),
    })

    # --- render PNG ---
    png_path = out_dir / "baselineB_examples.png"
    plt.figure(figsize=(16, 0.65 * len(table_df) + 1.2))
    plt.axis("off")

    tbl = plt.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()

    print("saved png:", png_path)


if __name__ == "__main__":
    main()