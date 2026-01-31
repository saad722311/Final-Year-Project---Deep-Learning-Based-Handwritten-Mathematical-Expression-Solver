# Phase B — Day 14  
## Rule-Based Structure Parsing: Dataset-Level Evaluation

### Objective
Evaluate the effectiveness and limits of **rule-based structure reconstruction** using **oracle symbol segmentation** on the CROHME 2023 test dataset, and determine whether further gains are possible without learning-based methods.

---

### Context (Progress So Far)
- Initial **CNN–LSTM** and **CNN–Transformer** end-to-end models failed to reliably predict full mathematical expressions.
- Introducing **symbol segmentation** significantly improved symbol-level accuracy, but **expression-level EM remained low** due to missing structural understanding.
- This motivated isolating the structure problem using **oracle segmentation** and rule-based parsing.

---

### What Was Done
- Evaluated rule-based structure reconstruction on the **full CROHME 2023 test set (2300 expressions)**.
- Used **oracle symbols** (ground-truth segmentation) to remove segmentation and classification errors.
- Implemented geometry-based rules for:
  - Fractions
  - Superscripts and subscripts
  - Operator limits (`\int`, `\sum`)
  - Distinguishing minus symbols from fraction bars

---

### Results

| Evaluation Stage | EM (%) | nEM (%) |
|------------------|--------|---------|
| Oracle symbols + basic linear rules | 8.74 | 16.83 |
| Oracle symbols + advanced rule-based structure parsing | **20.04** | **31.91** |

---

### Observations
- Clear improvement over the initial oracle-structure baseline.
- Simple and moderately complex expressions are reconstructed correctly.
- Major failure cases remain for:
  - Grouped superscripts/subscripts
  - Fractions combined with other operators
  - Large operators (integrals, summations)
  - Nested and multi-level structures
- Adding more heuristics results in **brittle, case-specific fixes** rather than consistent generalisation.

---

### Conclusion
Even with **perfect symbol segmentation**, rule-based structure parsing reaches a clear performance ceiling.  
These results demonstrate that **handcrafted rules are insufficient for robust mathematical structure understanding**, and strongly justify transitioning to a **learning-based structure prediction model** in the next phase.

This marks the end of rule-based methods as a baseline and the start of ML-driven structural inference.