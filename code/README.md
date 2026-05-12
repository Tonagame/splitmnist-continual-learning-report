# Code Included In This Report Repository

This folder contains the experiment code that was written or added during the project.

It is **not** a full copy of the original GMvandeVen repository.
The full upstream project is here:

https://github.com/GMvandeVen/continual-learning

## Files

| File | Purpose |
|---|---|
| `train_lsr_lite.py` | Experimental LSR-lite runner. Implements real replay, stored labels, stored teacher logits, stored feature vectors, optional Fourier loss, and optional Adaptive Stability Weighting. |
| `run_phase1_splitmnist_class_2000.ps1` | Runs the serious Split MNIST Class-CL 2000 experiment. |
| `run_phase2_splitmnist_domain_2000.ps1` | Runs the serious Split MNIST Domain-CL 2000 experiment. |
| `run_phase3_splitmnist_task_2000.ps1` | Runs the serious Split MNIST Task-CL 2000 experiment. |
| `phase1_summarize.py` | Generates Class-CL summary CSV, graphs, and report. |
| `phase2_summarize.py` | Generates Domain-CL summary CSV, graphs, and report. |
| `phase3_summarize.py` | Generates Task-CL summary CSV, graphs, and report. |
| `README_LSR.md` | Short implementation notes for LSR-lite. |
| `patches/eval_history_and_options.patch` | Patch showing the small changes made to the original repository for evaluation-history logging. |

## How To Use This Code

1. Clone the original repository:

   ```powershell
   git clone https://github.com/GMvandeVen/continual-learning.git
   ```

2. Copy the files from this `code/` folder into the root of the cloned repository.

3. Apply the patch in `code/patches/eval_history_and_options.patch`, or manually reproduce the changes.

4. Run one of the phase scripts from the repository root:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_phase1_splitmnist_class_2000.ps1
   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_phase2_splitmnist_domain_2000.ps1
   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_phase3_splitmnist_task_2000.ps1
   ```

## Important Protocol Notes

- Test data is used only for evaluation, not for training.
- LSR-lite builds its replay buffer from training data only.
- A-GEM and LSR-lite use the same memory budget where applicable: 100 samples per class.
- Class-CL evaluation does not use task identity.
- Task-CL evaluation keeps the original repository allowed-classes protocol.
- Generative Classifier was skipped where it was not cleanly supported.

