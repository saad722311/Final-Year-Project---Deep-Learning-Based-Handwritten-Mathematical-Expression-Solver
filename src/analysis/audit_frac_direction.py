from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

FRAC_RELS = {"FRAC_NUM", "FRAC_DEN"}

def is_frac_node(n: dict) -> bool:
    if not n.get("is_virtual"):
        return False
    lab = str(n.get("label", "")).upper()
    return "FRAC" in lab  # robust

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--gt_jsonl", required=True)
    args = ap.parse_args()

    def load(p):
        mp={}
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                o=json.loads(line)
                ui=o.get("ui") or Path(o.get("inkml_path","")).stem
                mp[ui]=o
        return mp

    pred = load(args.pred_jsonl)
    gt   = load(args.gt_jsonl)
    common = sorted(set(pred) & set(gt))
    print("intersection:", len(common))

    c = Counter()

    for ui in common:
        po = pred[ui]
        go = gt[ui]
        nodes = go.get("nodes", [])
        frac_idx = {i for i,n in enumerate(nodes) if is_frac_node(n)}
        if not frac_idx:
            continue

        edges = po.get("pred_edges", [])
        for s,t,r,sc in edges:
            if r not in FRAC_RELS:
                continue
            s=int(s); t=int(t)
            s_is = s in frac_idx
            t_is = t in frac_idx
            if s_is and (not t_is):
                c[f"{r}: FRAC->SYM"] += 1
            elif (not s_is) and t_is:
                c[f"{r}: SYM->FRAC"] += 1
            elif s_is and t_is:
                c[f"{r}: FRAC->FRAC"] += 1
            else:
                c[f"{r}: SYM->SYM"] += 1

    print("\n[FRAC direction counts]")
    for k,v in c.most_common():
        print(f"{k:20s} {v}")

if __name__ == "__main__":
    main()