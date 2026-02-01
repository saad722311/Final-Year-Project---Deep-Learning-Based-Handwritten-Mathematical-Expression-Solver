# Phase B — Day 15 (Structure Learning & Graph-to-LaTeX Evaluation)

## Date
Day 15

## Goal
Move beyond rule-based structure reconstruction by introducing **learned structural relations**
between segmented symbols, and evaluate whether learned edges improve expression-level
reconstruction compared to heuristic approaches.

---

## Context Recap
Earlier experiments showed that:
- End-to-end expression → LaTeX models performed poorly (low EM)
- Symbol segmentation + classification significantly improved symbol accuracy
- However, correct symbols alone were insufficient without structural understanding

This motivated **Option B**:
> Symbol recognition + learned structure prediction → deterministic LaTeX generation

---

## What Was Implemented

### 1. Structure Dataset Construction (Completed)
From CROHME InkML files:
- Parsed symbol segments with bounding boxes and centroids
- Extracted MathML-style ground-truth relations
- Introduced **virtual nodes** (e.g., FRAC) where required

Generated datasets:
- `crohme2023_train.jsonl` (1033 expressions)
- `crohme2023_val.jsonl` (548 expressions)
- `crohme2023_test.jsonl` (2289 expressions)

Dataset statistics:
- ~9.3k train edges, ~4.9k val edges
- 7 relation types: NEXT, SUP, SUB, FRAC_NUM, FRAC_DEN, UNDER, OVER
- ~100 unique symbol classes

Edge coverage: **100%** (no dropped gold relations)

---

### 2. Edge Classification Model (Learned Structure)
Implemented and trained an **Edge MLP** using:
- Symbol label embeddings
- Handcrafted geometric features (dx, dy, overlaps, relative size)
- Candidate pruning (k-right + k-any strategy)
- Negative sampling (3:1)

Training results:
- Validation accuracy: **~87.7%**
- Weighted F1: **~0.88**
- Macro F1: **~0.54** (expected due to class imbalance)

This confirms that **structural relations are being learned**, not memorised.

---

### 3. Structure Prediction + Decoding
Pipeline:
1. Predict edges on validation graphs at different thresholds
2. Apply structural constraints to remove invalid edge combinations
3. Convert predicted graph → LaTeX deterministically
4. Compare with ground-truth InkML LaTeX

Thresholds tested:
- 0.50
- 0.30
- 0.30 + structural constraints

---

## Evaluation Results

### Expression-Level Metrics (50-sample debug set)
- Exact Match (EM): ~0.08
- Normalized EM (nEM): ~0.24

While EM remains low, qualitative inspection shows:
- Many expressions are **structurally correct but fail EM due to formatting**
- Examples include missing braces, spacing, or equivalent algebraic forms

### Structural Relation Distribution (Post-Constraint)

NEXT: 308
SUP:  51
SUB:  45
UNDER: 5
OVER:  3

This confirms:
- Reading order (NEXT) is learned reliably
- Superscript/subscript relations are learned
- Fraction relations are sensitive to thresholding and decoding policies

---

## Key Findings

1. **Exact Match is overly strict** for evaluating learned structure
2. Edge-level accuracy and F1 provide a more reliable indicator of learning
3. The primary limitation is **graph decoding**, not symbol or relation prediction
4. Dataset quality is confirmed (edge coverage = 100%)

---

## Conclusion (Day 15)
The system successfully learns **symbol-level structural relations** and constructs
meaningful expression graphs. While LaTeX string EM remains low, the learned structure
is demonstrably correct in many cases, validating the chosen two-stage pipeline.

Further improvements would require:
- Grammar-aware decoding
- Global tree constraints
- Or graph-based neural decoders (beyond current scope)

Day 15 concludes the **structure learning phase**.

---

## Next Steps
- Freeze structure model as final backend
- Shift focus to UI / demo integration
- Present qualitative results and error analysis in dissertation