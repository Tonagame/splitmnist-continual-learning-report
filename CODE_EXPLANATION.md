# Code Explanation

## Current Rule

The project must not submit code copied from GitHub.
Therefore, the submission code is the clean-room implementation in:

`code/from_scratch/splitmnist_cl.py`

The earlier GMvandeVen-based runners are kept only as historical reference
material. They helped us understand the protocol and reference results, but the
new implementation is the code path to use for the final project.

## Main From-Scratch Components

### Dataset Construction

The script downloads MNIST through `torchvision` and builds Split MNIST itself.
The five contexts are:

```text
(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
```

Each dataset item returns:

```text
image, training_label, original_digit
```

This lets the code keep a clean distinction between:

- the label used for training,
- the original digit class,
- the replay-buffer key.

### Scenarios

`Class-CL`

- The model has 10 output units.
- There is no task identity at evaluation.
- Accuracy is measured over all 10 digit classes.

`Domain-CL`

- Each context is mapped to a shared binary label space.
- For example, in context `(4, 5)`, digit `4` maps to `0` and digit `5` maps to `1`.
- The model has 2 output units.

`Task-CL`

- The model has 10 output units for the normal methods.
- During evaluation, the code masks outputs to the two allowed classes of the active task.
- During sequential training, the supervised loss also uses only the active task's two allowed classes.
- Separate Networks uses one 2-output MLP per task.

## Neural Network

The default model is an MLP:

```text
784 input pixels
-> hidden layer
-> hidden layer
-> output logits
```

The default hidden layers are:

```text
400, 400
```

The model also exposes a `features(x)` function. LSR-lite uses this penultimate
feature vector for feature anchoring.

## Implemented Methods

### None

Sequential fine-tuning.
The model trains on context 1, then context 2, and so on.
There is no explicit protection against forgetting.

### Joint Training

The code concatenates all training contexts and trains on the union.
This is not a true continual-learning method; it is an upper-bound reference.

### EWC

Elastic Weight Consolidation is implemented with:

```text
loss = cross_entropy + 0.5 * lambda * sum(Fisher * (theta - theta_old)^2)
```

After each context, the script estimates a diagonal Fisher matrix from training
examples of that context. The Fisher estimate and parameter snapshot are then
used as a penalty during later contexts.

### LwF

Learning without Forgetting is implemented with a frozen teacher snapshot.
After each context, the current model is copied as the teacher.
During the next context, the student minimizes:

```text
cross_entropy(current labels) + lambda * KD(student logits, teacher logits)
```

The distillation loss uses temperature-scaled KL divergence.

### A-GEM

A-GEM uses a replay memory built only from training data.
For each update:

1. Compute the gradient on the current batch.
2. Compute the gradient on a replay batch.
3. If the current gradient conflicts with replay, project it:

```text
g_projected = g_current - dot(g_current, g_memory) / dot(g_memory, g_memory) * g_memory
```

Then the optimizer applies the projected gradient.

### Separate Networks

Separate Networks is implemented for Task-CL.
The code creates one MLP per task, each with two output units.
At evaluation time, the task identity selects the correct network.

### Generative Classifier

The clean-room Generative Classifier stores class statistics, not raw images.
For each class it keeps:

```text
count, sum, sum of squares
```

It then classifies by diagonal Gaussian log-likelihood.
This is a simple generative classifier and may not match the stronger generative
model from the paper without further tuning.

### LSR-lite

LSR-lite stores a balanced replay buffer from training data only.
For each stored example it keeps:

```text
image x
label y
teacher logits at insertion time
penultimate feature vector at insertion time
```

The training loss is:

```text
L_total =
  CE(current samples)
+ CE(replay samples)
+ lambda_kd * KD(replay logits, stored teacher logits)
+ lambda_feat * MSE(current replay features, stored features)
```

### LSR-lite + Fourier

This adds an auxiliary Fourier feature-spectrum loss:

```text
MSE(log(1 + abs(rFFT(current_features))),
    log(1 + abs(rFFT(stored_features))))
```

It does not replace real replay samples.

### LSR-lite + ASW

Adaptive Stability Weighting adjusts the KD and feature-anchor weights:

```text
adaptive_factor = old_signal / (new_loss + epsilon)
adaptive_factor = clamp(adaptive_factor, 0.5, 2.0)
```

Then:

```text
lambda_kd_eff = lambda_kd * adaptive_factor
lambda_feat_eff = lambda_feat * adaptive_factor
```

### LSR-lite + Fourier + ASW

This combines the Fourier auxiliary loss with ASW.
The Fourier weight stays fixed.

## Outputs

Each run writes:

- `summary.csv`
- `learning_curve.csv`
- one per-run learning-curve CSV
- one per-run metrics JSON file

The plotting script:

`code/from_scratch/plot_from_scratch_summary.py`

creates:

`from_scratch_final_accuracy.png`

## Smoke Tests

Smoke tests were run successfully on the NVIDIA RTX 3070 for:

- None
- EWC
- LwF
- A-GEM
- Generative Classifier
- LSR-lite + Fourier + ASW
- Separate Networks

These tests used only one iteration per context, so they verify code execution,
not final scientific accuracy.

## Task-CL Protocol Fix

The first from-scratch Task-CL run trained `None` and `EWC` with a full 10-class
cross-entropy loss. That did not match the Task-CL protocol, because Task-CL
uses task identity / allowed classes.

The code was corrected so the Task-CL supervised loss trains only on the active
task's two classes. After this fix:

- Task-CL `None` improved from 67.14% to 84.91%.
- Task-CL `EWC` improved from 64.75% to 88.06%.

This fixed most of the `None` gap. EWC still remains below the GMvandeVen result,
so the EWC implementation is marked as only a partial reproduction.

## Reproduction Target

The original GMvandeVen repository and the paper results are now reference
targets. To complete the reproduction under the new rule, the long experiments
must be run with the from-scratch code and compared against those references.
