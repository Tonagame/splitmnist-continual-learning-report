# Split MNIST Continual Learning Report

This repository contains a static GitHub Pages report for the Split MNIST continual-learning experiments.

Website, after GitHub Pages is enabled:

`https://tonagame.github.io/splitmnist-continual-learning-report/`

The report is written in Hebrew and summarizes experiments comparing:

- None
- EWC
- LwF
- A-GEM
- Separate Networks
- Generative Classifier, where supported
- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW
- Joint Training

## What Is Included

- `index.html` - the main Hebrew report page
- `styles.css` - page styling
- `assets/` - graphs, CSV summary, and Word report
- `assets/summary_hebrew_splitMNIST_2000.docx` - Hebrew Word report
- `assets/splitMNIST_2000_all_scenarios_summary.csv` - combined result table

## Where Are The Graphs?

The graph files are inside the `assets/` folder:

- `assets/all-methods-by-scenario.png`
- `assets/accuracy-heatmap.png`
- `assets/lsr-ablation-by-scenario.png`
- `assets/selected-learning-curves.png`

They are also embedded directly in the website.

### All Methods By Scenario

![All methods by scenario](assets/all-methods-by-scenario.png)

### Accuracy Heatmap

![Accuracy heatmap](assets/accuracy-heatmap.png)

### LSR Ablation By Scenario

![LSR ablation by scenario](assets/lsr-ablation-by-scenario.png)

### Selected Learning Curves

![Selected learning curves](assets/selected-learning-curves.png)

## Main Result Summary

| Scenario | None | Best non-Joint method | Joint |
|---|---:|---:|---:|
| Class-CL | 0.1982 | LSR-lite + Fourier + ASW: 0.9284 | 0.9815 |
| Domain-CL | 0.5435 | LSR-lite: 0.9651 | 0.9879 |
| Task-CL | 0.8765 | LwF: 0.9978 | 0.9981 |

## Main Conclusion

LSR-lite is most promising for the hardest setting: Class-Incremental Learning without task identity.

The core useful mechanism was:

- real replay samples from train data
- labels
- stored teacher logits
- stored penultimate feature vectors
- replay cross entropy
- logit distillation
- feature anchoring

Fourier and ASW were useful as ablations, but they were not the main reason the method worked.

## How To Enable GitHub Pages

Because this repository already has `index.html` at the repository root, use the root folder:

1. Open the repository on GitHub.
2. Go to `Settings -> Pages`.
3. Under `Build and deployment`, choose `Deploy from a branch`.
4. Choose branch `main`.
5. Choose folder `/ root`.
6. Click `Save`.

GitHub will publish the site after a short build.

## Notes

This repository contains only the static report site.
It does not include the full training code, datasets, Conda environment, or raw experiment folders.
