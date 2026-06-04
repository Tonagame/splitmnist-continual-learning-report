# Algorithmic Thinking

## Core Problem

The project studies **catastrophic forgetting**. A neural network learns several data contexts one after another. When it learns a new context, it can overwrite what it learned earlier.

Split MNIST makes this visible:

```text
context 1: digits 0 and 1
context 2: digits 2 and 3
context 3: digits 4 and 5
context 4: digits 6 and 7
context 5: digits 8 and 9
```

The algorithmic challenge is the **stability-plasticity tradeoff**:

- **Plasticity:** learn the new context.
- **Stability:** preserve old knowledge.

## Scenario Logic

| Scenario | Algorithmic meaning |
|---|---|
| Class-CL | The hardest case. No task identity is given. The model must choose among all 10 digits. |
| Domain-CL | The label space is shared, but data arrives through different contexts. |
| Task-CL | Task identity is known during evaluation, so the output space is restricted to the active task. |

## Method Logic

| Method | Main idea |
|---|---|
| None | Learn the current context only. This demonstrates forgetting. |
| Joint | Train on all data together. This is an upper bound, not a real continual-learning solution. |
| EWC | Estimate which parameters mattered for old contexts and penalize changing them. |
| LwF | Keep a frozen teacher model and distill its old predictions into the new model. |
| A-GEM | Keep replay memory and project gradients when the new update conflicts with old examples. |
| Separate Networks | Use one network per task. This works mainly when task identity is available. |
| H&T | Store real old examples plus labels, old logits, and feature vectors. |

## H&T Algorithmic Idea

H&T means **Haim and Tamir Hybrid Technique**.

For each saved replay example, it stores:

- image `x`
- label `y`
- teacher logits from the old model
- penultimate feature vector from the old model

The loss combines:

```text
current cross entropy
+ replay cross entropy
+ logit distillation
+ feature anchoring
+ optional Fourier auxiliary loss
```

This is a deliberate hybrid of replay and distillation ideas. It does not generate synthetic samples.

## Why This Is Algorithmic Thinking

The project does not only run commands. It compares different algorithmic strategies for the same problem:

- protect weights,
- preserve outputs,
- use replay memory,
- split models by task,
- preserve internal representations.

The results show that the scenario changes which strategy is effective. Class-CL needs stronger memory and representation stability, while Task-CL is much easier because the task identity is known.
