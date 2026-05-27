# Project Steps

This document lists the main steps of the project from setup to final reporting.

## 1. Environment Setup

- Created a local Windows setup on drive `E:`.
- Used Conda environment `continual`.
- Verified Python, Conda, Git, NVIDIA driver, and CUDA PyTorch.
- Confirmed the GPU as NVIDIA GeForce RTX 3070.

## 2. Reference Repository Setup

- Cloned the GMvandeVen continual-learning repository.
- Read the README and identified Split MNIST commands.
- Ran small smoke tests before long experiments.

## 3. Baseline Experiments

Ran Split MNIST baselines for:

- Class-CL
- Domain-CL
- Task-CL

Main baselines:

- None
- Joint Training

## 4. Continual-Learning Methods

Ran or implemented:

- EWC
- LwF
- A-GEM
- Separate Networks
- Generative Classifier where supported

## 5. LSR-lite Prototype

Implemented and tested:

- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW

The key LSR-lite mechanism was:

- real replay samples,
- stored labels,
- stored teacher logits,
- stored feature vectors.

## 6. Protocol Audit

Checked:

- no test data during training,
- replay buffers use train data only,
- Class-CL has no task identity,
- Task-CL keeps allowed-class evaluation,
- A-GEM and LSR use the same memory budget where applicable.

## 7. Serious 2000-Iteration Runs

Ran serious Split MNIST experiments with:

```text
contexts = 5
iters = 2000
batch = 128
acc-n = 1024
```

Ran for:

- Class-CL
- Domain-CL
- Task-CL

## 8. Clean-Room Code Requirement

After clarifying that copied GitHub code should not be submitted, a new independent implementation was created under:

[`code/from_scratch/`](../code/from_scratch/)

The code was refactored so each method family has its own file.

## 9. Graphs And Reports

Generated:

- final accuracy graphs,
- learning curve graphs,
- heatmap,
- paper-vs-reference-vs-from-scratch comparison,
- Word reports,
- GitHub Pages site.

## 10. Final GitHub Organization

The repository was reorganized into:

- [`README.md`](../README.md)
- [`index.html`](../index.html)
- [`docs/`](README.md)
- [`code/from_scratch/`](../code/from_scratch/)
- [`assets/`](../assets/)
