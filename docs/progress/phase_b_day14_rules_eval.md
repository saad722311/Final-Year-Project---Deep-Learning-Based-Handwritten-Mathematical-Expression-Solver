# Phase B — Day 14  
## Rule-Based Structure Parsing: Dataset-Level Evaluation

### Objective
Evaluate the effectiveness and limitations of **rule-based mathematical structure reconstruction** using **oracle symbol segmentation**, and determine whether further improvements are feasible without learning-based methods.

---

### Context (Progress So Far)
- Initial **CNN–LSTM** and **CNN–Transformer** end-to-end models failed to reliably predict full mathematical expressions.
- Introducing **symbol segmentation** significantly improved symbol-level accuracy, but **expression-level EM remained low** due to missing structural understanding.
- This motivated isolating the structure problem using **oracle segmentation**, removing segmentation and classification errors from the pipeline.

---

### What Was Done
- Evaluated a **rule-based structure parser** using **oracle symbols** on the CROHME 2023 dataset.
- Implemented geometry-based heuristics for:
  - Fraction detection
  - Superscripts and subscripts
  - Operator limits (`\int`, `\sum`)
  - Differentiation between minus symbols and fraction bars
- Performed **dataset-level evaluation** on train, validation, and test splits.

---

### Results

#### Oracle Symbols + Basic Linear Parsing (Baseline)
| Split | EM (%) | nEM (%) |
|------|--------|---------|
| Test | 8.74 | 16.83 |

#### Oracle Symbols + Advanced Rule-Based Structure Parsing
| Split | EM (%) | nEM (%) |
|------|--------|---------|
| Train | 16.65 | 34.07 |
| Validation | 24.14 | 36.76 |
| Test | **20.04** | **31.91** |

---

### Observations
- Rule-based structure parsing **more than doubles EM and nEM** compared to linear oracle baselines.
- Simple and moderately complex expressions are reconstructed correctly.
- Persistent failure cases include:
  - Grouped superscripts and subscripts
  - Fractions combined with other operators
  - Large operators (integrals, summations)
  - Nested and multi-level structures
- Adding more heuristics leads to **brittle, case-specific fixes** rather than consistent generalisation.

---

### Conclusion
Even with **perfect symbol segmentation**, rule-based structure parsing reaches a clear performance ceiling.  
These results confirm that **mathematical structure recognition cannot be robustly solved with handcrafted rules alone**.

This concludes the rule-based baseline and motivates the transition to a **learning-based structure prediction model**, which will be pursued in the next phase.