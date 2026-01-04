# MathWriting 2024 — Processed Dataset Snapshot

## Purpose
Freeze the processed MathWriting dataset state used for experiments (image → LaTeX).

## Location
- Dataset root: `03-development/datasets/mathwriting-2024/`
- Processed output: `03-development/datasets/mathwriting-2024/processed/`

## Processed structure
- `processed/train_images/` + `processed/train_labels.csv`
- `processed/valid_images/` + `processed/valid_labels.csv`
- `processed/test_images/`  + `processed/test_labels.csv`

Images are stored in the processed folders using sequential filenames:
- `000000.png, 000001.png, ...`

CSV schema:
- `filename` (e.g., `000123.png`)
- `label` (ground-truth LaTeX string)

## Sample counts (frozen)
- Train: **229,864**
- Valid: **15,674**
- Test:  **7,644**

> Counts correspond to rows in `*_labels.csv` and PNG files in `*_images/`.

## Reproducibility notes
- This snapshot is treated as fixed for reproducibility.
- Any future changes to preprocessing must be documented as a new dataset version.
