# Day 11 — Baseline Conclusions and Transition to Segmentation-Based Recognition

## Goal

The goal of Day 11 was to:

- Re-evaluate CNN–LSTM baselines after label normalization
- Confirm whether cleaned LaTeX labels lead to meaningful improvements in Exact Match (EM)
- Decide whether further end-to-end modeling is justified
- Begin transitioning toward a segmentation-based pipeline using CROHME InkML annotations

This day marks a methodological shift from pure end-to-end recognition to symbol-aware modeling.

---

## 1. Re-running Baselines with Cleaned Labels

After removing LaTeX math delimiters (`$...$`) and rebuilding the tokenizer, the CNN–LSTM baseline was retrained on the cleaned CROHME dataset.

### Training Summary (CNN–LSTM)

- Vocabulary size reduced: **84 → 83**
- Training loss converged smoothly
- Validation loss improved slightly compared to earlier runs

Best validation loss achieved:
Best val loss: 1.3473

However, loss improvements were **numerically small** and did not translate into meaningful EM gains.

---

## 2. Evaluation Results (CNN–LSTM)

### Validation Set

| Metric | Value |
|------|------|
| Exact Match (EM) | **0.00%** |
| Normalized EM | **0.47%** |
| Token Accuracy | **~65.7%** |
| Brace Balance | **~96.5%** |
| Avg Decoded Length | ~29 tokens |

### Test Set (Best Beam Configuration)

| Metric | Value |
|------|------|
| Exact Match (EM) | **0.17%** |
| Normalized EM | **0.46%** |
| Token Accuracy | **~61.3%** |
| Brace Balance | **~94.8%** |

### Qualitative Observations

- Predictions are syntactically valid LaTeX
- Many symbols are locally correct
- Entire expressions are semantically incorrect
- Small structural errors cause EM = 0
- Beam search improves plausibility but not correctness

---

## 3. Key Diagnosis

Despite label normalization:

- Token-level learning is strong
- Structural reconstruction remains unreliable
- EM remains extremely low

This confirms that **label formatting was not the primary bottleneck**.

The fundamental issue is that the model attempts to learn:

- symbol segmentation
- spatial relationships
- LaTeX grammar
- long-range dependencies

**simultaneously**, from a limited dataset (~9k samples).

---

## 4. Decision: End-to-End Models Are Data-Limited on CROHME

Based on results from:

- CNN–LSTM
- CNN–Transformer (overfit + baseline tests)
- Clean vs unclean labels

We conclude:

- Further end-to-end tuning is unlikely to yield significant EM gains
- CROHME is too small for holistic image-to-LaTeX learning
- Token accuracy does not imply expression correctness

This aligns with observations from the CROHME competition literature.

---

## 5. Segmentation Insight from CROHME InkML

CROHME provides **InkML files with stroke-level annotations**, including:

- Explicit `traceGroup` elements
- Ground-truth symbol labels per stroke group
- Links to MathML structural elements

Example (simplified):

<traceGroup>
  <annotation type="truth">+</annotation>
  <traceView traceDataRef="6"/>
  <traceView traceDataRef="7"/>
</traceGroup>

This means:
	•	Symbol segmentation is already provided
	•	No heuristic segmentation is required
	•	Symbol-level datasets can be constructed directly

---

## 6. Why Segmentation Is Expected to Improve EM

Segmentation decomposes the problem into simpler tasks:

Task A — Symbol Recognition
	•	Input: cropped symbol image (from strokes)
	•	Output: symbol class (e.g., x, +, π)
	•	This is a well-studied, high-accuracy task

Task B — Structural Reconstruction
	•	Use spatial relations (above, below, inside)
	•	Build expression tree
	•	Deterministically convert to LaTeX

This removes the need for the model to implicitly infer structure.

---

## 7. Planned Segmentation-Based Pipeline

InkML
 ├─ traceGroups (symbols)
 │   ├─ strokes → symbol image → symbol classifier
 │   └─ symbol label
 └─ spatial relations
     └─ expression tree → LaTeX

Expected benefits:
	•	Higher symbol accuracy
	•	Reduced structural ambiguity
	•	Substantially higher EM

---

## 8. Next Steps (Day 12 Plan)
        1. Parse InkML files programmatically
        2. Extract symbol-level stroke groups
        3. Render individual symbol images
        4. Build a symbol classification dataset
        5. Train a CNN-based symbol recognizer
        6. Report symbol-level accuracy
        7. Integrate symbol predictions into expression reconstruction

---

## Conclusion

Day 11 establishes that:
	• Cleaned labels alone do not solve EM failure
	• End-to-end models are insufficient for CROHME scale
	• CROHME’s stroke-level annotations make segmentation feasible and well-motivated

This justifies a clear transition to a segmentation-aware recognition framework, consistent with prior CROHME competition approaches and expected to significantly improve exact-match performance.


