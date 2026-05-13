# Reproduction Report

## Short Summary

The project reproduced the main Split MNIST continual-learning trends:

- Class-CL is the hardest scenario because the model must classify among all 10 digits without task identity.
- Domain-CL is easier than Class-CL but still suffers from forgetting.
- Task-CL is much easier because task identity restricts evaluation to the active digit pair.
- Joint Training remains the upper-bound reference.
- A-GEM and LwF reproduce the expected trend reasonably well.
- EWC is implemented, but our clean-room version is still a partial reproduction in Task-CL.

## Main Comparison Graph

![Paper vs GMvandeVen vs from-scratch](../assets/paper_vs_gmvandeven_vs_from_scratch.png)

The graph compares:

1. Published paper values.
2. Our earlier local run of the GMvandeVen reference code.
3. Our independent clean-room implementation.

## Key Result Files

- `../assets/paper_vs_gmvandeven_vs_from_scratch.csv`
- `../assets/from_scratch_classic_no_lsr_2000_summary.csv`
- `../assets/splitMNIST_2000_all_scenarios_summary.csv`

## What Matched Well

- Class-CL `None` is near the paper's roughly 20% result.
- Joint Training is near the paper's roughly 98-99% result.
- Task-CL `None` became close to the paper after fixing the Task-CL training protocol.
- LwF and A-GEM follow the expected pattern: they help more when the scenario allows their assumptions to work.

## What Did Not Match Perfectly

EWC is the main weak point. After correcting Task-CL training, EWC improved from about 65% to about 88%, but the GMvandeVen reference run was near 99.5%.

The likely reason is that our EWC implementation is simpler:

- Fisher estimation is simplified.
- Fisher scaling and normalization may differ.
- Consolidation details may not match the reference implementation exactly.

This is documented as a partial reproduction, not hidden.

## LSR-lite Result

LSR-lite is our experimental bonus method. It is not part of the paper.

It combines:

- real replay samples from train data,
- stored labels,
- LwF-style teacher logits,
- feature anchoring,
- optional Fourier loss,
- optional Adaptive Stability Weighting.

It performed especially well in Class-CL and Domain-CL, where catastrophic forgetting is more visible.

Detailed explanation:

`LSR_LITE_EXPLANATION.md`
