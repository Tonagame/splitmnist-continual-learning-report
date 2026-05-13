# LSR-lite Explanation And Ablations

## What Is LSR-lite?

LSR-lite is our experimental continual-learning prototype.

The name means:

```text
Latent Stability Replay - lite
```

The goal is to reduce catastrophic forgetting by keeping a small, fair memory of
old training examples and preserving both the model's old output behavior and
its internal feature representation.

## Is LSR-lite A Hybrid Of A-GEM, LwF, And Generative Classifier?

Almost, but with an important correction.

LSR-lite is best described as a hybrid of:

| Source idea | What LSR-lite borrows |
|---|---|
| A-GEM / replay methods | A small memory buffer of old real training examples |
| LwF | Knowledge distillation from stored teacher logits |
| Representation regularization | Feature anchoring using stored penultimate feature vectors |

It is **not really a Generative Classifier hybrid**, because it does not train a
generative model and it does not generate synthetic old samples.

A more accurate sentence is:

> LSR-lite combines exemplar replay, LwF-style distillation, and feature
> anchoring. It is conceptually related to memory/replay methods such as A-GEM,
> but unlike a Generative Classifier it keeps real old examples instead of
> learning a generative model.

## What Is Stored In Memory?

For each saved replay example, LSR-lite stores:

- image `x`
- label `y`
- teacher logits at insertion time
- penultimate feature vector at insertion time

The buffer is class-balanced and uses the same budget as A-GEM in our
comparison:

```text
100 samples per class
```

The buffer is built from train data only. Test data is never used for training.

## Loss Function

The basic LSR-lite loss is:

```text
L_total =
  L_CE_current
+ L_CE_replay
+ lambda_kd   * L_KD_logits
+ lambda_feat * L_feature_anchor
```

Where:

- `L_CE_current` learns the current context.
- `L_CE_replay` relearns labels of old replay examples.
- `L_KD_logits` keeps the current model close to the old teacher predictions.
- `L_feature_anchor` keeps internal feature vectors close to their stored values.

## Ablation 1: LSR-lite

Plain LSR-lite uses:

- real replay samples
- labels
- stored teacher logits
- stored feature vectors
- cross entropy
- logit distillation
- feature anchoring

This is the core method.

## Ablation 2: LSR-lite + Fourier

This variant adds an auxiliary Fourier / spectral feature regularization term.

Important:

Fourier does **not** replace replay.
It does **not** store FFT signatures instead of images.
It is only an extra loss term on the feature vectors.

The added term is:

```text
L_Fourier =
MSE(log(1 + abs(rFFT(current_features))),
    log(1 + abs(rFFT(stored_features))))
```

The full loss becomes:

```text
L_total =
  L_CE_current
+ L_CE_replay
+ lambda_kd   * L_KD_logits
+ lambda_feat * L_feature_anchor
+ lambda_fft  * L_Fourier
```

In our main run:

```text
lambda_fft = 0.05
```

## Ablation 3: LSR-lite + ASW

ASW means:

```text
Adaptive Stability Weighting
```

The idea is to adapt the strength of the stability losses based on the ratio
between old/replay pressure and new/current learning pressure.

The adaptive factor is:

```text
adaptive_factor = old_loss / (new_loss + epsilon)
adaptive_factor = clamp(adaptive_factor, 0.5, 2.0)
```

Then:

```text
lambda_kd_eff   = lambda_kd   * adaptive_factor
lambda_feat_eff = lambda_feat * adaptive_factor
```

Defaults:

```text
lambda_kd = 1.0
lambda_feat = 0.5
temperature = 2
epsilon = 1e-8
```

## Ablation 4: LSR-lite + Fourier + ASW

This combines:

- real replay
- labels
- stored teacher logits
- stored feature vectors
- Fourier auxiliary feature loss
- Adaptive Stability Weighting

The Fourier weight remains fixed; ASW only adapts the distillation and feature
anchor weights.

## Main 2000-Iteration Results

These results are from the earlier controlled 2000-iteration experiments.
They are useful as experimental results for LSR-lite, but the classic baseline
methods also have a newer from-scratch reproduction path in `code/from_scratch/`.

| Scenario | LSR-lite | LSR-lite + Fourier | LSR-lite + ASW | LSR-lite + Fourier + ASW | Best LSR variant |
|---|---:|---:|---:|---:|---|
| Class-CL | 92.43% | 92.64% | 92.41% | 92.84% | Fourier + ASW |
| Domain-CL | 96.51% | 96.27% | 95.74% | 95.93% | LSR-lite |
| Task-CL | 99.33% | 99.26% | 99.19% | 99.40% | Fourier + ASW |

## What We Learned From The Ablations

1. The core replay mechanism matters most.
2. Fourier helps slightly in Class-CL, but does not dominate.
3. ASW alone does not always improve performance.
4. Fourier + ASW gives the best Class-CL LSR result in our runs.
5. Domain-CL preferred plain LSR-lite.
6. Task-CL is already easy because task identity is available, so all LSR
   variants are close to Joint.

## How To Explain It Simply

LSR-lite tries to solve forgetting by asking the model to remember old examples
in three ways:

1. **Remember the label:** replay cross entropy.
2. **Remember the old output behavior:** LwF-style logit distillation.
3. **Remember the internal representation:** feature anchoring.

Fourier and ASW are optional additions:

- Fourier checks whether the feature vector keeps a similar spectral structure.
- ASW changes the stability strength dynamically during training.

## Honest Limitation

LSR-lite is our experimental prototype, not a method from the paper.
It should be presented as a new ablation/prototype compared against the paper
and GMvandeVen reference methods, not as an official reproduced method.
