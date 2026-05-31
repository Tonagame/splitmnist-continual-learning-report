# Testing By Stage

This document explains how the project was checked at each stage.

## 1. Environment Validation

Checked:

- Python runs correctly.
- Conda environment exists.
- Git is available.
- PyTorch imports successfully.
- CUDA is available.
- GPU name is NVIDIA GeForce RTX 3070.

Purpose:

To make sure long experiments run on GPU and not accidentally on CPU.

## 2. Smoke Tests

Before long runs, short runs were executed with very small settings, for example:

```text
iters = 1
batch = 16
acc-n = 64
```

Purpose:

To catch import errors, CUDA errors, bad method arguments, and result-writing problems before spending hours on full runs.

## 3. Baseline Validation

Checked that:

- None runs sequentially.
- Joint Training uses all contexts together.
- results are saved to `summary.csv`.
- intermediate evaluations are saved to `learning_curve.csv`.

## 4. Method Validation

Each method was tested separately:

- EWC: Fisher estimation and penalty.
- LwF: teacher snapshot and distillation.
- A-GEM: replay memory and gradient projection.
- Separate Networks: one model per task.
- LSR-lite: replay buffer with logits/features.

## 5. Protocol Validation

Important checks:

- Test data is not used for training.
- Replay memory is built only from train data.
- Class-CL evaluation does not use task identity.
- Class-CL final evaluation uses all 10 classes.
- Task-CL uses allowed-class masking, as required by the protocol.
- A-GEM and LSR-lite use the same default memory budget: 100 samples per original digit class.

## 6. Long-Run Validation

For 2000-iteration runs, checked:

- every method produced logs,
- failed methods did not stop the whole experiment,
- `run_status.csv` recorded success/failure,
- final accuracy was written,
- learning curves were written,
- graphs were generated after aggregation.

## 7. Refactor Validation

After splitting the code into method files, the following checks were run:

- Python compile check on the refactored files.
- Smoke tests on CUDA / RTX 3070 for representative methods:
  - None
  - EWC
  - LwF
  - A-GEM
  - LSR-lite + Fourier + ASW
  - Joint
  - Separate Networks

## 8. Documentation Validation

Checked that:

- README links point to the correct docs.
- GitHub Pages uses root `index.html`.
- graphs are stored under `assets/`.
- code explanation and project explanation are linked from README and website.
