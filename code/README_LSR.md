# LSR-lite Prototype

This repository copy includes a small experimental runner for **LSR-lite** on Split MNIST Class-CL:

```powershell
& "E:\conda-envs\continual\python.exe" train_lsr_lite.py --experiment=splitMNIST --scenario=class --contexts=5 --iters=100 --batch=128 --acc-n=1024 --results-dir="E:\Codex\continual-learning-setup\continual-learning\results\lsr-lite-class" --model-dir="E:\Codex\continual-learning-setup\continual-learning\results\models"
```

The Fourier ablation is enabled only with:

```powershell
& "E:\conda-envs\continual\python.exe" train_lsr_lite.py --experiment=splitMNIST --scenario=class --contexts=5 --iters=100 --batch=128 --acc-n=1024 --fourier --results-dir="E:\Codex\continual-learning-setup\continual-learning\results\lsr-lite-class" --model-dir="E:\Codex\continual-learning-setup\continual-learning\results\models"
```

## What Was Implemented

`train_lsr_lite.py` is intentionally separate from the original `main.py` method switches. It reuses the repository's data loading, classifier definition, optimizer style, and evaluation helpers, but keeps the prototype isolated.

LSR-lite uses a real exemplar replay buffer. For each stored sample it keeps:

- image tensor `x`
- class label `y`
- teacher logits at insertion time
- penultimate feature vector at insertion time

During training it optimizes:

- cross entropy on current samples
- cross entropy on replay samples
- logit distillation on replay samples
- feature anchoring on replay samples

The default buffer budget is `100` exemplars per class, matching the existing A-GEM default budget used in these quick Split MNIST runs.

## Fourier Ablation

The Fourier variant does **not** replace replay with FFT signatures. It uses the same real exemplar replay buffer as LSR-lite.

When `--fourier` is passed, the runner adds an auxiliary feature-spectrum anchoring term:

- compute `rfft` magnitudes over the current penultimate feature vector for replay samples
- compare those magnitudes with the stored penultimate feature spectrum
- add the loss with `--fourier-weight`, default `0.1`

This is deliberately an auxiliary regularizer, not the core memory mechanism.

## Outputs

The quick Class-CL comparison artifacts are saved under:

```text
E:\Codex\continual-learning-setup\continual-learning\results\lsr-lite-class
```

Key files:

- `summary.csv`
- `splitMNIST_class_lsr_comparison.png`
- `logs\splitMNIST_class_lsr_lite.log`
- `logs\splitMNIST_class_lsr_lite_fourier.log`

## Current Scope

This is a prototype for Split MNIST Class-CL. It does not add bottleneck adapters yet, because adapters were optional and not needed to validate the core replay-plus-anchoring idea. It also does not modify the repository's original EWC, LwF, A-GEM, Joint, or Generative Classifier implementations.
