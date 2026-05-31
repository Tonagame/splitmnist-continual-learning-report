# Algorithmic Thinking: What The Human Did

## Project Problem

The project studies **catastrophic forgetting** in continual learning.

Instead of training on all MNIST digits at once, the model sees Split MNIST as a sequence of five contexts:

```text
context 1: digits 0 and 1
context 2: digits 2 and 3
context 3: digits 4 and 5
context 4: digits 6 and 7
context 5: digits 8 and 9
```

The core challenge is the **stability-plasticity tradeoff**:

- **Plasticity:** the model must learn the new context.
- **Stability:** the model must preserve useful knowledge from old contexts.

## Human Decisions In The Project

The human project work was not only running code. The important human decisions were:

- choosing the paper: *Three types of incremental learning*;
- choosing Split MNIST as the reproducible dataset;
- deciding to compare Class-CL, Domain-CL, and Task-CL;
- deciding to use the local NVIDIA RTX 3070 GPU;
- checking whether results were fair and comparable;
- requiring no test data during training;
- requiring true Class-CL evaluation without task identity;
- requiring the same memory budget for A-GEM and LSR-lite variants;
- deciding to remove the simplified generative-classifier experiment from the final method set;
- deciding to add LSR-lite as an experimental bonus prototype.

## Scenario Logic

| Scenario | What It Tests | Why It Matters |
|---|---|---|
| Class-CL | The model must choose among all 10 digits without task identity. | This is the hardest and most realistic setting. |
| Domain-CL | The model uses a shared label space while data arrives in different contexts. | This tests robustness to changing input distributions. |
| Task-CL | The model is told which task is active during evaluation. | This is easier because the allowed classes are restricted. |

## Method Logic

| Method | Algorithmic Idea |
|---|---|
| None | Train only on the current context. This shows forgetting directly. |
| Joint | Train on all data together. This is an upper bound, not a true continual-learning method. |
| EWC | Penalize changes to parameters that were important for previous contexts. |
| LwF | Keep a frozen teacher model and distill old behavior into the new model. |
| A-GEM | Use replay memory and project gradients when the new update conflicts with old examples. |
| Separate Networks | Use one model per task. This mainly fits Task-CL. |
| LSR-lite | Store real old examples, labels, teacher logits, and feature vectors. |

## LSR-lite Human Design

LSR-lite means:

```text
Latent Stability Replay - lite
```

The method was designed as a memory-and-stability prototype.

For each replay example, LSR-lite stores:

- image `x`,
- label `y`,
- teacher logits at insertion time,
- penultimate feature vector at insertion time.

The training loss combines:

```text
current cross entropy
+ replay cross entropy
+ logit distillation
+ feature anchoring
+ optional Fourier auxiliary loss
```

The key human idea was:

> If the model sees old examples again, remembers its old predictions, and keeps its old internal representation stable, it should forget less.

## Why This Counts As Algorithmic Thinking

The project compares different algorithmic strategies for the same forgetting problem:

- protect weights,
- preserve old outputs,
- use replay memory,
- split models by task,
- preserve internal representations.

The results show that the best strategy depends on the scenario. Class-CL needs stronger memory and representation stability, while Task-CL is easier because task identity is available.
