# What Was Implemented vs What Was Reused

This project used the original GMvandeVen `continual-learning` repository as the base implementation.

Original repository:

https://github.com/GMvandeVen/continual-learning

## Short Answer

We did **not** reimplement the classic paper methods from scratch.

The existing repository already implemented:

- None / sequential baseline
- Joint Training
- EWC
- LwF
- A-GEM
- Separate Networks
- Generative Classifier where supported

We used those implementations by running the repository with the correct command-line flags.

What we **did implement** for this project:

- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW
- evaluation-history CSV logging
- phase runner scripts
- summary scripts
- result graphs
- GitHub Pages report

## Why The GitHub Code Looks Like Runner Code

This report repository is not a full fork of GMvandeVen's repository.

It contains:

1. The code written for this project.
2. The scripts used to run the experiments.
3. A patch showing the small changes made to the original repository.
4. Documentation and results.

It does **not** copy the entire upstream research repository, because the upstream project is already public and large.
The classic methods live in the upstream repository.

## Classic Methods: Reused From The Original Repository

| Method | Implemented by us? | Where it comes from |
|---|---:|---|
| None | No | Original training loop in GMvandeVen repository |
| Joint Training | No | Original `main.py` / dataset-combination logic |
| EWC | No | Original `models/cl/continual_learner.py` and `train/train_task_based.py` |
| LwF | No | Original distillation/replay logic in `main.py`, `train/train_task_based.py`, and `models/cl/continual_learner.py` |
| A-GEM | No | Original replay-buffer and inequality-gradient logic |
| Separate Networks | No | Original `models/separate_classifiers.py` |
| Generative Classifier | No | Original generative-classifier training code, where supported |

Useful upstream implementation locations:

- `main.py` - parses method flags and connects options to model behavior.
- `train/train_task_based.py` - main task-based training loop.
- `models/cl/continual_learner.py` - EWC, replay, distillation, A-GEM-related continual-learning logic.
- `models/separate_classifiers.py` - Separate Networks implementation.
- `models/define_models.py` - creates normal classifiers, separate classifiers, and generative models.
- `params/options.py` and `params/param_values.py` - method flags and default hyperparameters.

## New Method Implemented In This Project: LSR-lite

The main new method is implemented in:

`code/train_lsr_lite.py`

LSR-lite stores a real replay buffer from training data only.
For each stored sample it keeps:

- image `x`
- label `y`
- teacher logits at insertion time
- penultimate feature vector at insertion time

During training, it combines:

```text
L_total =
  L_CE_current
+ L_CE_replay
+ lambda_kd   * L_KD_logits
+ lambda_feat * L_feature_anchor
+ lambda_fft  * L_Fourier_optional
```

## LSR-lite + Fourier

This is an ablation implemented in `code/train_lsr_lite.py`.

It adds an auxiliary Fourier / spectral regularization term on feature vectors.
It does **not** replace real replay samples.

## LSR-lite + ASW

This is also implemented in `code/train_lsr_lite.py`.

ASW means Adaptive Stability Weighting.
It dynamically adjusts the distillation and feature-anchor weights:

```text
adaptive_factor = old_loss / (new_loss + epsilon)
adaptive_factor = clamp(adaptive_factor, 0.5, 2.0)

lambda_kd_eff   = lambda_kd   * adaptive_factor
lambda_feat_eff = lambda_feat * adaptive_factor
```

## Evaluation Logging Implemented In This Project

The patch:

`code/patches/eval_history_and_options.patch`

adds:

- `--eval-history-file`
- `--eval-history-method`
- a CSV logging callback for learning curves

The learning-curve CSV contains:

```csv
method,iteration,context,accuracy
```

## Experiment Automation Implemented In This Project

These scripts were written for the serious 2000-iteration experiments:

- `code/run_phase1_splitmnist_class_2000.ps1`
- `code/run_phase2_splitmnist_domain_2000.ps1`
- `code/run_phase3_splitmnist_task_2000.ps1`

They run the selected methods with the same settings:

- Split MNIST
- 5 contexts
- 2000 iterations
- batch size 128
- acc-n 1024
- buffer budget 100 where applicable

## Summary And Graph Code Implemented In This Project

These scripts collect outputs and generate reports:

- `code/phase1_summarize.py`
- `code/phase2_summarize.py`
- `code/phase3_summarize.py`

They create:

- `summary.csv`
- `learning_curve.csv`
- final accuracy bar graphs
- learning curve graphs
- Markdown phase reports

## Bottom Line

The classic continual-learning algorithms were reused from the original research repository.

The new contribution in this project was:

1. implementing LSR-lite and its ablations,
2. auditing the evaluation protocol,
3. running controlled experiments,
4. generating graphs and reports,
5. publishing the results and code explanations on GitHub Pages.

