# Code Explanation

This document explains the code added for the Split MNIST continual-learning project.

## Original Repository

The project started from the GMvandeVen continual-learning repository:

https://github.com/GMvandeVen/continual-learning

That repository already supports many continual-learning methods, including:

- None / baseline sequential training
- Joint Training
- EWC
- LwF
- A-GEM
- Separate Networks
- Generative Classifier in supported settings

The code added in this project was intentionally kept small and separate.
The goal was to avoid rewriting the original repository.

## Evaluation-History Logging

The original repository already evaluated accuracy during training, but the project needed raw learning-curve data saved to CSV.

The patch file:

`code/patches/eval_history_and_options.patch`

adds two command-line options:

- `--eval-history-file`
- `--eval-history-method`

It also adds a callback that writes rows like:

```csv
method,iteration,context,accuracy
LSR-lite,100,1,0.99
```

For Task-CL, the callback preserves the original Task-CL evaluation behavior by using allowed classes for the current task.
For Class-CL, no task identity is used.

## LSR-lite

The main experimental code is:

`code/train_lsr_lite.py`

LSR-lite is a prototype continual-learning method based on real replay.

For old examples, it stores:

- image `x`
- label `y`
- teacher logits at insertion time
- penultimate feature vector at insertion time

During later contexts, each training step can mix the current batch with replay samples from the buffer.

The total loss is:

```text
L_total =
  L_CE_current
+ L_CE_replay
+ lambda_kd   * L_KD_logits
+ lambda_feat * L_feature_anchor
+ lambda_fft  * L_Fourier_optional
```

### Cross Entropy

Cross entropy trains the model to predict the correct label for current samples and replay samples.

### Logit Distillation

Logit distillation compares the current model's logits on a replay sample with the old teacher logits saved when the sample entered memory.

This helps preserve old decision behavior.

### Feature Anchoring

Feature anchoring compares the current penultimate feature vector with the stored old feature vector.

This encourages the model to keep a similar internal representation for old examples.

### Fourier Auxiliary Loss

The Fourier option adds a small auxiliary loss on the spectrum of the feature vector.

It is an ablation, not the main memory mechanism.
Real replay samples are still used.

### Adaptive Stability Weighting

ASW dynamically changes the effective weights of the distillation and feature losses:

```text
adaptive_factor = old_loss / (new_loss + epsilon)
adaptive_factor is clamped between 0.5 and 2.0
```

Then:

```text
lambda_kd_eff   = lambda_kd   * adaptive_factor
lambda_feat_eff = lambda_feat * adaptive_factor
```

The goal is to increase stability when replay performance worsens and reduce it when the replay loss is already low.

## Phase Runner Scripts

The PowerShell scripts automate the serious 2000-iteration experiments.

### Phase 1

`code/run_phase1_splitmnist_class_2000.ps1`

Runs Split MNIST Class-CL.
This is the hardest scenario because there is no task identity during evaluation.

### Phase 2

`code/run_phase2_splitmnist_domain_2000.ps1`

Runs Split MNIST Domain-CL.

### Phase 3

`code/run_phase3_splitmnist_task_2000.ps1`

Runs Split MNIST Task-CL.
This keeps the repository's original Task-CL allowed-classes evaluation protocol.

## Summary Scripts

The summary scripts collect result files and generate:

- `summary.csv`
- `learning_curve.csv`
- final accuracy bar graphs
- learning-curve graphs
- short Markdown reports

Files:

- `code/phase1_summarize.py`
- `code/phase2_summarize.py`
- `code/phase3_summarize.py`

## What The Code Demonstrates

The main algorithmic idea is the stability-plasticity tradeoff.

The model must learn new contexts while keeping old knowledge.
LSR-lite addresses this by storing real old examples and preserving both:

- the old output behavior through logits
- the old internal representation through feature anchoring

The experiments showed that this approach is especially useful for Class-CL, where there is no task identity at evaluation time.
