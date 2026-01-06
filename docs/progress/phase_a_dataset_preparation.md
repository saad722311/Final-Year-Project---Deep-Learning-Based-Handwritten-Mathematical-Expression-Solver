# Phase A — Dataset Preparation and Freeze

**Phase:** A  
**Objective:** Prepare, clean, and freeze datasets for reproducible model training and evaluation.

---

## Datasets used

### 1) CROHME (TC11_CROHME23)
- **Source:** CROHME 2013, CROHME 2019, OffHME (training)
- **Evaluation split:** CROHME 2023 (validation & test)
- **Modality:** Offline handwritten images (PNG)
- **Labels:** Ground-truth LaTeX extracted from InkML

### 2) MathWriting 2024
- **Source:** MathWriting 2024 dataset
- **Modality:** Offline handwritten images (PNG)
- **Labels:** Ground-truth LaTeX provided by dataset

---

## Preprocessing steps (CROHME)

- Converted InkML annotations to LaTeX labels
- Matched InkML files with corresponding PNG images
- Removed invalid samples:
  - Missing PNG or missing LaTeX
  - Non-expression / metadata labels (e.g., *Closest Strk*)
- Normalised LaTeX labels (whitespace cleanup)
- Reorganised data into MathWriting-style structure:
  - Sequential image filenames (`000000.png`, `000001.png`, …)
  - CSV-based label files

### Processed structure
processed/
train_images/ + train_labels.csv
valid_images/ + valid_labels.csv
test_images/  + test_labels.csv

---

## Dataset snapshot (frozen)

### CROHME (processed)
- **Train:** 9,363 samples
- **Validation:** 428 samples
- **Test:** 1,757 samples

Training data sources:
- CROHME 2013
- CROHME 2019
- OffHME

Validation and test data:
- CROHME 2023

### MathWriting 2024 (processed)
- **Train:** 229,864 samples
- **Validation:** 15,674 samples
- **Test:** 7,644 samples

---

## Reproducibility and version control

- Raw and processed datasets are **not committed** to version control
- All preprocessing steps are fully reproducible via scripts
- Dataset states are documented and frozen via snapshot markdown files:
  - `docs/data/crohme23_processed.md`
  - `docs/data/mathwriting2024_processed.md`

This ensures:
- Reproducibility
- Clean repository structure
- Compliance with dataset licensing constraints

---

## Phase A outcome

✅ Datasets prepared, cleaned, and frozen  
✅ Preprocessing scripts committed  
✅ Dataset statistics documented  
✅ Ready to proceed with Phase B (baseline recognition pipeline)