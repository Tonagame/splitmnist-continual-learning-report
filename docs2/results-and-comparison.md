# Results And Comparison

This document summarizes the main Split MNIST results and compares them with the paper and the reference-code runs.

## Paper Reference

The project is based on:

van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). **Three types of incremental learning**. *Nature Machine Intelligence*.

Paper link:

https://www.nature.com/articles/s42256-022-00568-3

Reference repository:

https://github.com/GMvandeVen/continual-learning

## Important Caveat

The paper reports averages and SEM over multiple seeds. Our local long runs are mostly single-seed runs, so the comparison is approximate and should be interpreted as a reproduction of trends rather than a full statistical reproduction.

## Main Comparison Graphs

### Paper vs Reference Code vs From-Scratch Code

![Paper vs GMvandeVen vs from-scratch](../assets/paper_vs_gmvandeven_vs_from_scratch.png)

Raw CSV:

[`../assets/paper_vs_gmvandeven_vs_from_scratch.csv`](../assets/paper_vs_gmvandeven_vs_from_scratch.csv)

### All Methods By Scenario

![All methods by scenario](../assets/all-methods-by-scenario.png)

Raw CSV:

[`../assets/splitMNIST_2000_all_scenarios_summary.csv`](../assets/splitMNIST_2000_all_scenarios_summary.csv)

### Accuracy Heatmap

![Accuracy heatmap](../assets/accuracy-heatmap.png)

### H&T Ablation Graph

![H&T ablation by scenario](../assets/h-and-t-ablation-by-scenario.png)

### Selected Learning Curves

![Selected learning curves](../assets/selected-learning-curves.png)

## Common Method Comparison

| Scenario | Method | Paper % | Our / reference trend |
|---|---|---:|---|
| Class-CL | None | 19.89 | Very close to paper; shows strong forgetting. |
| Class-CL | EWC | 20.64 | Similar low-performance trend in Class-CL. |
| Class-CL | LwF | 21.89 | Similar low-performance trend in Class-CL. |
| Class-CL | A-GEM | 65.10 | Helpful, but our run was lower than the paper. |
| Class-CL | Joint | 98.17 | Very close to paper; upper-bound behavior. |
| Domain-CL | None | 60.13 | Same general range, with some local-run variation. |
| Domain-CL | EWC | 63.03 | Similar range. |
| Domain-CL | LwF | 71.18 | Our/reference runs were somewhat higher. |
| Domain-CL | A-GEM | 87.67 | Close to paper. |
| Domain-CL | Joint | 98.59 | Very close to paper. |
| Task-CL | None | 84.32 | Close after correcting the Task-CL protocol. |
| Task-CL | EWC | 99.06 | EWC remains a partial reproduction in the from-scratch code. |
| Task-CL | LwF | 99.60 | Close to paper. |
| Task-CL | A-GEM | 98.54 | Close to paper. |
| Task-CL | Separate Networks | 99.57 | Close to paper. |
| Task-CL | Joint | 99.67 | Close to paper. |

## H&T Results

H&T is not from the paper. It is our experimental prototype.

| Scenario | Best H&T Variant | Accuracy % | Interpretation |
|---|---|---:|---|
| Class-CL | H&T + Fourier + ASW | 92.84 | Strong result in the hardest scenario. |
| Domain-CL | H&T | 96.51 | Best non-Joint H&T-family result for Domain-CL. |
| Task-CL | H&T + Fourier + ASW | 99.40 | Strong, but Task-CL is already easier due to task identity. |

## Main Interpretation

- Class-CL is the hardest scenario.
- Joint Training remains the upper bound.
- A-GEM is the strongest classic method in Class-CL among the reproduced classic baselines.
- H&T performs strongly because it stores richer replay signals.
- Task-CL results are generally high because task identity is available.
- EWC is implemented, but the from-scratch version remains a partial reproduction compared with the strongest reference result.
