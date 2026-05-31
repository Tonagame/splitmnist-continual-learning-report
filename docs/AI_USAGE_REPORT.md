# AI Usage Report

This document explains how AI assistance was used during the project.

## Why AI Was Used

AI was used as a development assistant, debugging partner, and documentation helper. The project still required human decisions about the paper, dataset, methods, experiment settings, and final interpretation.

## How AI Helped By Project Stage

### 1. Setup

AI helped plan the local setup:

- verify Python, Conda, Git, CUDA, and PyTorch,
- create a clean environment,
- keep the project on drive `E:`,
- check GPU availability.

### 2. Reading The Reference Repository

AI helped interpret the GMvandeVen repository structure and identify how to run small Split MNIST experiments.

This was used for reference behavior only. The final submitted implementation is the clean-room code under:

[`code/from_scratch/`](../code/from_scratch/)

### 3. Experiment Planning

AI helped turn the project into clear phases:

- baseline runs,
- selected continual-learning methods,
- LSR-lite ablations,
- Class-CL 2000-iteration run,
- Domain-CL 2000-iteration run,
- Task-CL 2000-iteration run.

### 4. Debugging And Protocol Checks

AI helped check important risks:

- no test data during training,
- replay buffers are built from train data only,
- Class-CL has no task identity,
- Task-CL uses allowed classes,
- A-GEM and LSR-lite use comparable memory budgets.

### 5. Clean-Room Implementation

When the requirement became clear that copied GitHub code should not be submitted, AI helped design and implement a new independent version of the code.

The final code was split into:

- [`splitmnist_cl.py`](../code/from_scratch/splitmnist_cl.py)
- [`core.py`](../code/from_scratch/core.py)
- [`methods/`](../code/from_scratch/methods/)

### 6. Reports And Graphs

AI helped generate:

- CSV summaries,
- accuracy graphs,
- learning-curve graphs,
- report-style Markdown documentation,
- GitHub Pages HTML,
- Markdown documentation.

## What AI Did Not Replace

AI did not replace the scientific responsibility of the project. The final report still states limitations:

- not all paper datasets were reproduced,
- most long runs are single-seed runs,
- EWC is a partial reproduction,
- LSR-lite is our experimental prototype and not a method from the paper.

## Useful AI Links

- ChatGPT / Codex: https://chatgpt.com/
- GitHub Pages documentation: https://docs.github.com/en/pages
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Original reference repository: https://github.com/GMvandeVen/continual-learning

## Summary

AI was used to accelerate setup, planning, debugging, coding, and documentation. The final repository keeps the process transparent by separating paper results, reference-code runs, and our own from-scratch implementation.
