# AI Documentation

This file summarizes how AI assistance was used during the project.

It combines the AI plan summary, AI usage report, and AI work log into one document.

## Why AI Was Used

AI was used as:

- a setup assistant,
- a debugging partner,
- a planning partner,
- a coding assistant,
- a documentation helper.

AI helped speed up the work, but it did not replace the scientific responsibility of the project.

## Main AI-Assisted Plans

### 1. Local Setup

AI helped plan how to:

- check Python, Conda, Git, CUDA, and PyTorch;
- create a clean Conda environment;
- keep the project on drive `E:`;
- verify the NVIDIA RTX 3070 GPU;
- run a first Split MNIST smoke test.

### 2. Baseline Experiments

AI helped plan baseline runs:

- None,
- Joint Training,
- Class-CL,
- Domain-CL,
- Task-CL,
- logs, CSVs, and graphs.

### 3. Continual-Learning Methods

AI helped organize the selected methods:

- EWC,
- LwF,
- A-GEM,
- Separate Networks where appropriate.

The project later removed the simplified generative-classifier experiment from the submitted method set.

### 4. H&T Prototype

AI helped design and implement the H&T prototype:

- replay buffer with real old training samples,
- labels,
- teacher logits,
- penultimate feature vectors,
- feature anchoring,
- Fourier ablation,
- ASW ablation.

### 5. Serious 2000-Iteration Runs

AI helped structure the long experiments:

- `contexts=5`,
- `iters=2000`,
- `batch=128`,
- `acc-n=1024`,
- logs for every method,
- final accuracy graphs,
- learning curves,
- summaries.

### 6. Independent Implementation

When the requirement became clear that copied GitHub code should not be submitted, AI helped create the independent implementation under:

[`../code/from_scratch/`](../code/from_scratch/)

This implementation separates:

- dataset and evaluation logic,
- model code,
- method-specific code,
- graph and CSV generation.

## Human Decisions

The student made the main project decisions:

- use Split MNIST;
- run on RTX 3070;
- compare Class-CL, Domain-CL, and Task-CL;
- add H&T, Fourier, and ASW;
- switch from reference-code runs to from-scratch implementation;
- keep limitations visible;
- remove methods that were not cleanly comparable.

## Transparency

The project separates three result sources:

- paper values,
- reference-code runs,
- our own from-scratch implementation.

This separation is important for the defense because it shows what was reproduced, what was used as reference, and what was newly implemented.

## Useful Links

- ChatGPT / Codex: https://chatgpt.com/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- GitHub Pages documentation: https://docs.github.com/en/pages
- Original reference repository: https://github.com/GMvandeVen/continual-learning
