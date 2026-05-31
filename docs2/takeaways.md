# Takeaways

## Main Lesson

The main lesson is that continual learning is not one problem. The difficulty changes depending on whether the model receives task identity.

- **Class-CL** is hardest because the model must choose among all 10 digits.
- **Domain-CL** is easier because the label space is shared.
- **Task-CL** is easiest because task identity restricts the allowed classes.

## What We Learned About Forgetting

The `None` baseline shows catastrophic forgetting clearly. It trains on each new context and has no mechanism to protect old knowledge.

In Class-CL, this makes performance collapse close to chance-level behavior for the full 10-class problem.

## What We Learned About Classic Methods

EWC, LwF, and A-GEM represent three different strategies:

- **EWC:** protect important weights.
- **LwF:** preserve old output behavior.
- **A-GEM:** use replay memory and avoid harmful gradients.

A-GEM helped more than EWC and LwF in Class-CL, but it was still far from Joint Training.

## What We Learned About LSR-lite

LSR-lite was strongest in the hardest setting, Class-CL.

Its core idea is not only to store old examples. It stores:

- the old image,
- the correct label,
- the old model's logits,
- the old model's internal feature vector.

This gives the model three reminders:

1. remember the correct label,
2. remember old output behavior,
3. remember old internal representation.

## Fourier And ASW

Fourier and ASW were useful ablations, but they were not the main reason LSR-lite worked.

- Fourier adds a spectral feature regularization term.
- ASW adapts the stability loss weights dynamically.

The main strength still came from replay plus distillation plus feature anchoring.

## Best Practical Interpretation

For a defense, the clean explanation is:

> The project reproduced key Split MNIST trends from the paper and then tested an experimental replay-and-stability method, LSR-lite. The results show that richer replay signals help most when task identity is unavailable.

## Limitations

Important limitations:

- most long runs are single-seed runs;
- the full paper includes more datasets and methods than this project reproduces;
- EWC is a partial reproduction in the from-scratch code;
- LSR-lite is an experimental prototype, not a method from the original paper.

## Future Work

Good next steps would be:

- run multiple seeds,
- tune EWC more carefully,
- test LSR-lite on CIFAR-100 or TinyImageNet,
- compare different replay-buffer sizes,
- evaluate whether Fourier and ASW help on harder image datasets.
