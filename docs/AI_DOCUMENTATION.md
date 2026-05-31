# AI Documentation

This document combines the AI plan summary, AI usage report, and AI work log for the Split MNIST continual-learning project.

The purpose is not to submit the full chat transcript. Instead, this file gives a clear and defensible summary of how AI assistance was used during the project, what plans were created, what work was done, and which decisions remained human project decisions.

## Why AI Was Used

AI was used as a development assistant, debugging partner, planning partner, and documentation helper.

It helped with:

- setting up the local Windows / Conda / CUDA workflow,
- interpreting the GMvandeVen reference repository,
- planning smoke tests and long experiments,
- writing PowerShell runners,
- debugging experiment failures,
- designing the LSR-lite prototype,
- checking protocol issues such as test-data leakage and Class-CL evaluation,
- creating CSV summaries and graphs,
- writing the GitHub Pages report,
- reorganizing the final GitHub repository.

AI did not replace the scientific responsibility of the project. The student still chose the paper, dataset, main experiment direction, methods to compare, and final interpretation.

## AI-Assisted Plans

### Plan 1: Local Setup

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

### Plan 2: Baseline Experiments

Goal:

- run Split MNIST baselines.

Plan:

1. Run None and Joint.
2. Run for Class-CL, Domain-CL, and Task-CL.
3. Save results.
4. Generate comparison graph.
5. Report commands and accuracies.

### Plan 3: Continual-Learning Methods

Goal:

- compare selected continual-learning methods.

Plan:

1. Run EWC, LwF, and A-GEM.
2. Run Separate Networks only where appropriate.
3. Keep settings identical.
4. Save logs, CSV files, and graphs.

### Plan 4: LSR-lite Prototype

Goal:

- test an experimental memory-based method.

Plan:

1. Store real replay examples from train data.
2. Store labels.
3. Store teacher logits.
4. Store penultimate feature vectors.
5. Train with current cross entropy, replay cross entropy, KD loss, and feature anchoring.
6. Add Fourier and ASW as separate ablations.

### Plan 5: Serious 2000-Iteration Runs

Goal:

- run serious experiments for Class-CL, Domain-CL, and Task-CL.

Plan:

1. Use `contexts=5`, `iters=2000`, `batch=128`, `acc-n=1024`.
2. Save each scenario in a separate results folder.
3. Continue if one method fails.
4. Save logs and learning curves.
5. Generate final graphs and reports.

### Plan 6: Clean-Room Implementation

Goal:

- satisfy the requirement that the project code should not be copied from GitHub.

Plan:

1. Implement Split MNIST data construction independently.
2. Implement the MLP model independently.
3. Implement each method independently.
4. Keep the original repository only as a reference target.
5. Document limitations honestly.

### Plan 7: GitHub Finalization

Goal:

- make the GitHub repository understandable and ready for defense.

Plan:

1. Create a GitHub Pages report.
2. Add Markdown documentation.
3. Add report-style Markdown pages.
4. Add graphs and CSV files.
5. Add code explanations.
6. Reorganize docs and code folders.
7. Add a submission checklist.

## How AI Helped By Project Stage

### 1. Setup

AI helped plan and validate the local setup:

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

## Important Human Decisions

The project direction and requirements were provided by the student:

- use Split MNIST,
- run on RTX 3070,
- compare Class-CL, Domain-CL, and Task-CL,
- add LSR-lite, Fourier, and ASW ablations,
- switch from using the original repository directly to implementing methods independently,
- remove the simplified generative-classifier experiment from the submitted method set,
- keep limitations visible instead of hiding them.

## What AI Did Not Replace

AI did not replace the scientific responsibility of the project. The final report still states limitations:

- not all paper datasets were reproduced,
- most long runs are single-seed runs,
- EWC is a partial reproduction,
- LSR-lite is our experimental prototype and not a method from the paper.

## Transparency Note

The early phase used the GMvandeVen repository as a runnable reference. Later, because the assignment required implementing the methods ourselves, a clean-room implementation was added under:

`code/from_scratch/`

The documentation separates:

- paper values,
- reference-code runs,
- our own implementation.

This separation is important for the defense.

## Useful Links

- ChatGPT / Codex: https://chatgpt.com/
- GitHub Pages documentation: https://docs.github.com/en/pages
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Original reference repository: https://github.com/GMvandeVen/continual-learning

## Summary

AI was used to accelerate setup, planning, debugging, coding, and documentation. The final repository keeps the process transparent by separating paper results, reference-code runs, and our own from-scratch implementation.
