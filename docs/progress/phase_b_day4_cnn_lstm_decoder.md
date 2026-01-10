# Phase B — Day 4: End-to-End Training Pipeline (CNN–LSTM)

**Phase:** B  
**Day:** 4  
**Objective:** Train the first full end-to-end handwritten mathematical expression recognition baseline on CROHME using a CNN–LSTM architecture.

---

## What was done

- Integrated the full training pipeline:
  - CNN encoder
  - LSTM decoder with attention
  - Character-level tokenizer
- Implemented a **picklable batch collator** to support `num_workers > 0` on macOS
- Resolved multiprocessing issues caused by lambda-based `collate_fn`
- Successfully trained the model on the CROHME processed dataset

---

## Training setup

- **Dataset:** CROHME (processed, offline)
- **Architecture:** CNN encoder + LSTM decoder with attention
- **Tokenizer:** Character-level LaTeX tokenizer
- **Device:** Apple MPS
- **Epochs:** 5 (sanity training run)
- **Batch size:** 8
- **num_workers:** 2 (using picklable collator)

---

## Training results (summary)

- Training loss decreased steadily across epochs
- Validation loss consistently improved
- Best validation loss achieved: **2.3198**

This confirms:
- The end-to-end model is correctly wired
- Forward + backward passes are stable
- Data loading, batching, and padding work as intended

---

## Outcome

✅ End-to-end CNN–LSTM baseline successfully trained  
✅ Training pipeline validated and stable  
✅ Ready for evaluation and decoding experiments  

---

## Next step (Day 5)

- Implement inference decoding (greedy decoding)
- Run qualitative evaluation on validation samples
- Prepare metrics for dissertation reporting