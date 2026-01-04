# CROHME (TC11_CROHME23) — Processed Dataset Snapshot

## Purpose
Freeze the dataset state used for Phase A baseline experiments (offline image → LaTeX).

## Location
- Raw dataset root: `03-development/datasets/TC11_CROHME23/`
- Processed output: `03-development/datasets/TC11_CROHME23/processed/`

## Processed structure
- `processed/train_images/` + `processed/train_labels.csv`
- `processed/valid_images/` + `processed/valid_labels.csv`
- `processed/test_images/`  + `processed/test_labels.csv`

Images are copied into the processed folders and renumbered sequentially:
- `000000.png, 000001.png, ...`

CSV schema:
- `filename` (e.g., `000123.png`)
- `label` (ground-truth LaTeX string)

## Sample counts (frozen)
- Train: **9363**
- Valid: **428**
- Test:  **1757**

> Counts correspond to the number of rows in `*_labels.csv` and the number of PNGs in `*_images/`.

## Data sources / split policy
### Train (historical + auxiliary)
Training images are taken from the available `IMG/train/` subsets in the dataset distribution:
- CROHME 2013 (train)
- CROHME 2019
- OffHME

### Validation & Test (CROHME 2023)
Evaluation is performed on CROHME 2023:
- Valid: `CROHME2023_val`
- Test: `CROHME2023_test`

## Label extraction
Labels are extracted from InkML annotation fields (ground-truth) and normalised (whitespace normalisation).

## Label filtering rule
Non-expression / metadata annotations are excluded from the processed dataset.
Examples of filtered labels include:
- `Closest Strk` / `Closest Stroke`
- other non-mathematical, stroke/segmentation metadata

A sample is kept only if:
- it has a matching PNG image, and
- the extracted label appears to be a mathematical expression (contains LaTeX/math-like tokens or digits),
- and it does not match known metadata label patterns.

## Reproducibility notes
- This snapshot should not be modified after baseline training begins.
- Any future changes to preprocessing or filtering must be documented as a new dataset version.
