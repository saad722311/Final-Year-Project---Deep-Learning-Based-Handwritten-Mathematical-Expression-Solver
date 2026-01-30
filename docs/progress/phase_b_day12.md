# Phase B — Day 12 (Symbol Segmentation + Symbol Classifier Baseline)

## Date
Day 12

## Goal
Move beyond end-to-end “whole expression → LaTeX” (low EM) by using CROHME InkML stroke-level segmentation to:
1) Generate symbol-level cropped PNGs + labels  
2) Train a symbol classifier baseline  
3) Evaluate symbol-level accuracy on valid/test splits  
4) Attempt a first “expression reconstruction” sanity test from predicted symbols

---

## What was achieved

### 1) InkML segmentation extraction works
- Parsed CROHME InkML files successfully and extracted symbol-level segments using `<traceGroup>` → `<traceView traceDataRef=...>` mappings.
- Verified that each symbol segment corresponds to one or multiple traces (strokes), and that the extracted list matches expectations for a sample expression.

Example (sanity parse):
- UI: `form_001_Eq1`
- Extracted 6 symbol segments: `3`, `+`, `z`, `2`, `-`, `8`
- Trace IDs were correctly grouped (e.g., `+` used multiple traces).


### 2) Symbol dataset generated + validated (Step 4 sanity check)
Generated dataset:
- Train: `symbols_train_images/` + `symbols_train_labels.csv`
- Valid: `symbols_valid_images/` + `symbols_valid_labels.csv`
- Test:  `symbols_test_images/` + `symbols_test_labels.csv`

Sanity-check confirmed:
- PNG crops are **not blank**
- Symbols are **visible** and generally **not cut off**
- Dataset loading works using `SymbolDataset`

Dataset check output:
- Train symbols: **10711**
- Classes: **99**
- Sample item: `form_001_E1__sym000.png` → label `3`

---

## 3) Symbol classifier baseline trained (TinySymbolCNN)
Trained using:
python3 -m src.symbol_recognition.train --config configs/symbol_cnn_baseline.yaml
After fixing config to use the valid split correctly, training achieved:

Validation
	•	Best valid accuracy: 0.8213 (~82%)

Training dynamics
	•	Valid accuracy steadily improved across epochs
	•	Loss decreased as expected (no obvious training collapse)

## 4) Proper test evaluation added (handling unseen labels)

Problem found

Running evaluation on the raw test set caused a crash:
	•	KeyError: '\mu'

Reason:
	•	Some symbols exist in test that are not present in the training label vocabulary (e.g., \mu, \exists).

Fix implemented

Updated evaluation flow to support unseen labels handling:
	•	Option: --unseen skip
	•	Saves:
	•	unseen labels list
	•	filtered CSV containing only “seen” labels

Test evaluation result (seen-only):
	•	Test accuracy: 0.8160 (~81.6%)
	•	Kept: 22874 samples
	•	Dropped: 40 samples
	•	Unseen labels: ['\\exists', '\\mu']

Commands used:

python3 -m src.symbol_recognition.eval \
  --images_dir 03-development/datasets/TC11_CROHME23/segmented/symbols_test_images \
  --labels_csv 03-development/datasets/TC11_CROHME23/segmented/symbols_test_labels.csv \
  --ckpt results/crohme_symbol_cnn_baseline_day12/best.pt \
  --out results/crohme_symbol_cnn_baseline_day12/test_top_confusions.csv \
  --unseen skip \
  --unseen_out results/crohme_symbol_cnn_baseline_day12/unseen_test_labels.txt \
  --filtered_csv_out 03-development/datasets/TC11_CROHME23/segmented/symbols_test_labels_seenonly.csv \
  --device auto --batch_size 256 --num_workers 0

## 5) Confusion hotspots identified

Top confusions (examples):
	•	\times → x
	•	| → 1
	•	( → 1
	•	z → 2
	•	x → 2
	•	\pi → x

These indicate common handwriting ambiguity + visually similar classes and suggest the classifier is learning meaningful patterns but still struggles with:
	•	symbol shape similarity (x vs \times, 1 vs |)
	•	small symbols or weak crops
	•	Greek letters vs Latin look-alikes

## 6) Expression-level reconstruction attempt (current limitation)

We attempted a simple expression prediction by:
	1.	Segmenting symbols from InkML
	2.	Classifying each symbol
	3.	Concatenating outputs into a LaTeX-like string

Result example:
	•	UI: form_001_Eq1
	•	GT: \frac{3+z^2}{8}
	•	PRED: \sqrt \sqrt \sqrt^{\sqrt} \sqrt^{\sqrt}
	•	EM: 0

Key finding

High symbol accuracy does NOT directly translate into correct expression LaTeX, because the missing piece is structure:
	•	reading order (left-to-right)
	•	superscript/subscript relations
	•	fraction numerator/denominator grouping
	•	“inside” relations (sqrt contents, brackets, etc.)