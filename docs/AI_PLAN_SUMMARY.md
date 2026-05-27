# AI Plan Summary

This document summarizes the planning work done with AI during the project.

The purpose is not to submit the full chat transcript. Instead, this file gives a clear equivalent of the major plans and decisions that guided the work.

## Plan 1: Local Setup

Goal:

- run the GMvandeVen continual-learning repository locally,
- use the NVIDIA RTX 3070 GPU,
- avoid using drive `C:` and work under drive `E:`.

Plan:

1. Check Python, Conda, Git, NVIDIA driver, and PyTorch CUDA.
2. Create a clean Conda environment.
3. Install CUDA-enabled PyTorch.
4. Clone the repository.
5. Run a small Split MNIST smoke test.

## Plan 2: Baseline Experiments

Goal:

- run Split MNIST baselines.

Plan:

1. Run None and Joint.
2. Run for Class-CL, Domain-CL, and Task-CL.
3. Save results.
4. Generate comparison graph.
5. Report commands and accuracies.

## Plan 3: Continual-Learning Methods

Goal:

- compare selected methods.

Plan:

1. Run EWC, LwF, A-GEM.
2. Run Generative Classifier where supported.
3. Run Separate Networks only where appropriate.
4. Keep settings identical.
5. Save logs, CSV files, and graphs.

## Plan 4: LSR-lite Prototype

Goal:

- test an experimental memory-based method.

Plan:

1. Store real replay examples from train data.
2. Store labels.
3. Store teacher logits.
4. Store penultimate feature vectors.
5. Train with current cross entropy, replay cross entropy, KD loss, and feature anchoring.
6. Add Fourier and ASW as separate ablations.

## Plan 5: Serious 2000-Iteration Runs

Goal:

- run serious experiments for Class-CL, Domain-CL, and Task-CL.

Plan:

1. Use `contexts=5`, `iters=2000`, `batch=128`, `acc-n=1024`.
2. Save each scenario in a separate results folder.
3. Continue if one method fails.
4. Save logs and learning curves.
5. Generate final graphs and reports.

## Plan 6: Clean-Room Implementation

Goal:

- satisfy the requirement that the project code should not be copied from GitHub.

Plan:

1. Implement Split MNIST data construction independently.
2. Implement the MLP model independently.
3. Implement each method independently.
4. Keep the original repository only as a reference target.
5. Document limitations honestly.

## Plan 7: GitHub Finalization

Goal:

- make the GitHub repository understandable and ready for defense.

Plan:

1. Create a GitHub Pages report.
2. Add Markdown documentation.
3. Add Word reports.
4. Add graphs and CSV files.
5. Add code explanations.
6. Reorganize docs and code folders.
7. Add a submission checklist.
