# Paper Results vs Our Results

This document compares our Split MNIST results with Table 2 from van de Ven, Tuytelaars & Tolias, "Three types of incremental learning", Nature Machine Intelligence 2022.

Source: https://www.nature.com/articles/s42256-022-00568-3

Important caveat: the paper reports mean +/- SEM over 20 random seeds. Our project results are single local runs, so this is an approximate comparison, not a statistically matched reproduction.

![Paper vs ours](assets/paper_vs_ours_splitMNIST_common_methods.png)

## Common Methods

| Scenario | Method | Paper % | Paper SEM | Our % | Our - Paper | Note |
|---|---|---:|---:|---:|---:|---|
| Class-CL | None | 19.89 | 0.02 | 19.82 | -0.07 |  |
| Class-CL | EWC | 20.64 | 0.52 | 27.42 | 6.78 |  |
| Class-CL | LwF | 21.89 | 0.32 | 20.27 | -1.62 |  |
| Class-CL | A-GEM | 65.10 | 3.64 | 55.80 | -9.30 |  |
| Class-CL | Generative Classifier | 93.82 | 0.06 |  |  | not run / failed in our run |
| Class-CL | Joint | 98.17 | 0.04 | 98.15 | -0.02 |  |
| Domain-CL | None | 60.13 | 1.66 | 54.35 | -5.78 |  |
| Domain-CL | EWC | 63.03 | 1.58 | 64.64 | 1.61 |  |
| Domain-CL | LwF | 71.18 | 1.42 | 78.63 | 7.45 |  |
| Domain-CL | A-GEM | 87.67 | 1.33 | 88.19 | 0.52 |  |
| Domain-CL | Joint | 98.59 | 0.05 | 98.79 | 0.20 |  |
| Task-CL | None | 84.32 | 0.99 | 87.65 | 3.33 |  |
| Task-CL | EWC | 99.06 | 0.15 | 99.52 | 0.46 |  |
| Task-CL | LwF | 99.60 | 0.03 | 99.78 | 0.18 |  |
| Task-CL | A-GEM | 98.54 | 0.10 | 99.04 | 0.50 |  |
| Task-CL | Separate Networks | 99.57 | 0.03 | 99.74 | 0.17 |  |
| Task-CL | Joint | 99.67 | 0.03 | 99.81 | 0.14 |  |

## Our New LSR-lite Results

LSR-lite is not a method from the paper. It was implemented in this project, so it should be compared against paper methods only as a new experimental prototype.

| Scenario | Best LSR variant | Accuracy % | Comparable paper reference |
|---|---|---:|---|
| Class-CL | LSR-lite + Fourier + ASW | 92.84 | Below paper BI-R 94.41 and Generative Classifier 93.82, above DGR 90.35, ER 88.79 and A-GEM 65.10 |
| Domain-CL | LSR-lite | 96.51 | Above paper DGR 95.57 and ER 93.75, below paper BI-R 97.26 |
| Task-CL | LSR-lite + Fourier + ASW | 99.40 | Close to paper LwF 99.60 / Separate Networks 99.57 / Joint 99.67 |

## Main Interpretation

- Our reproduction of the simple baselines and Joint is close to the paper, especially in Class-CL and Joint.
- Some classic methods differ because the paper reports 20-seed averages, while our runs are single local runs.
- Our A-GEM in Class-CL was lower than the paper: 55.80% vs 65.10%.
- Our LwF and EWC in Domain/Task were somewhat higher than the paper values.
- LSR-lite is not in the paper, but it performed strongly: especially Class-CL and Domain-CL.
- In Class-CL, our best LSR variant reached 92.84%, which is much higher than paper A-GEM and ER, close to Generative Classifier, but still below paper BI-R.
