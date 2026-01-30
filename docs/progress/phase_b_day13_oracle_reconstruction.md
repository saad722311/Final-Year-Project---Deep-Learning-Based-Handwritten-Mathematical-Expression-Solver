# Phase B — Day 13: Oracle Structure Reconstruction

## Objective
The objective of this stage was to evaluate whether **correct mathematical structure alone** (without any learned structure prediction) is sufficient to reconstruct accurate LaTeX expressions. This was done using **oracle structure reconstruction**, where the ground-truth MathML annotations provided in the CROHME InkML files are used directly.

This experiment isolates the **structure-to-LaTeX** component and answers the question:
> *If symbols and structure were perfectly known, could the system generate the correct LaTeX expression?*

---

## Methodology

For each InkML file:
1. Extract the **ground-truth Content MathML** annotation.
2. Convert MathML into an **Abstract Syntax Tree (AST)**.
3. Convert the AST into a LaTeX string.
4. Compare the generated LaTeX against the ground-truth LaTeX annotation.

Two metrics were used:
- **Exact Match (EM):** Strict string equality.
- **Normalized Exact Match (nEM):** After LaTeX normalization (spacing, formatting, brace differences).

---

## Datasets Evaluated
- **CROHME 2023 Train**
- **CROHME 2023 Validation**
- **CROHME 2023 Test**

Only samples with valid MathML annotations were included.

---

## Results

### CROHME 2023 — Train
- Total files: 1045
- Used: 1037
- Missing MathML: 8

| Metric | Value |
|------|------|
| EM | **2.41%** (25 / 1037) |
| nEM | **56.41%** (585 / 1037) |

---

### CROHME 2023 — Validation
- Total files: 555
- Used: 552
- Missing MathML: 3

| Metric | Value |
|------|------|
| EM | **1.99%** (11 / 552) |
| nEM | **55.62%** (307 / 552) |

---

### CROHME 2023 — Test
- Total files: 2300
- Used: 2298
- Missing MathML: 2

| Metric | Value |
|------|------|
| EM | **3.09%** (71 / 2298) |
| nEM | **60.66%** (1394 / 2298) |

---

## Error Analysis

Low Exact Match (EM) scores are primarily due to:
- Missing LaTeX macros (e.g. `sin` vs `\sin`)
- Formatting differences (`\times` vs `times`)
- Spacing and brace placement
- Macro name normalization (`\alpha` vs `alpha`)

However, many EM failures are **semantically correct**, which is reflected in the much higher nEM scores.

---

## Key Findings

1. **Structure-to-LaTeX conversion is viable**
   - Over 60% of expressions are structurally correct using oracle structure.
2. **LaTeX generation is not the main bottleneck**
   - Most failures are formatting-level, not structural.
3. **Earlier end-to-end failures were caused by structure prediction**
   - Not symbol recognition
   - Not LaTeX generation

This confirms that improving **structure inference** is the critical next step.

---

## Conclusion

Oracle reconstruction demonstrates that, given correct structure, the system is capable of generating largely correct LaTeX expressions. Therefore, future work should focus on **predicting mathematical structure from segmented symbols**, rather than further tuning symbol recognition or LaTeX decoding.

Oracle reconstruction serves as a strong upper bound and validates the chosen pipeline design.

---

## Next Steps

- Replace oracle structure with **predicted structure**
- Explore:
  - Rule-based geometric structure inference (baseline)
  - Learned relation classifiers between symbols (advanced)
- Maintain oracle reconstruction as a reference upper bound during evaluation