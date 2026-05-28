# Split MNIST Continual Learning Reproduction

This repository is the final GitHub submission package for a project on catastrophic forgetting and continual learning with Split MNIST.

It includes:

- a GitHub Pages report,
- Markdown documentation,
- clean-room Python code,
- graphs,
- CSV result files,
- Word reports,
- reflective writing,
- AI usage documentation,
- a video checklist/link placeholder.

Published website after GitHub Pages is enabled:

[`https://tonagame.github.io/splitmnist-continual-learning-report/`](https://tonagame.github.io/splitmnist-continual-learning-report/)

## Submission Checklist

| Rule / requirement | Link |
|---|---|
| GitHub site includes everything | [`index.html`](index.html) |
| Full submission checklist | [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md) |
| Algorithmic thinking | [`docs/ALGORITHMIC_THINKING.md`](docs/ALGORITHMIC_THINKING.md) |
| Project stages | [`docs/PROJECT_STEPS.md`](docs/PROJECT_STEPS.md) |
| Testing and validation by stage | [`docs/TESTING_BY_STAGE.md`](docs/TESTING_BY_STAGE.md) |
| AI plan summaries | [`docs/AI_PLAN_SUMMARY.md`](docs/AI_PLAN_SUMMARY.md) |
| AI usage report | [`docs/AI_USAGE_REPORT.md`](docs/AI_USAGE_REPORT.md) |
| AI work log | [`docs/AI_WORK_LOG.md`](docs/AI_WORK_LOG.md) |
| Reflective writing / takeaways | [`docs/takeaways.md`](docs/takeaways.md) |
| Video link / video instructions | [`docs/VIDEO.md`](docs/VIDEO.md) |
| LSR-lite animation for video recording | [`lsr_animation.html`](lsr_animation.html) |
| Clean-room code | [`code/from_scratch/`](code/from_scratch/) |
| Code explanation Word report | [`assets/code_explanation_full.docx`](assets/code_explanation_full.docx) |
| Full Word report with graphs | [`assets/full_project_explanation_with_all_graphs.docx`](assets/full_project_explanation_with_all_graphs.docx) |

## Main Code

The submitted implementation is:

[`code/from_scratch/`](code/from_scratch/)

Important files:

- [`splitmnist_cl.py`](code/from_scratch/splitmnist_cl.py) - command-line entry point and method dispatch.
- [`core.py`](code/from_scratch/core.py) - dataset, MLP model, replay buffer, evaluation, result writing.
- [`methods/`](code/from_scratch/methods/) - method-specific implementations.

Implemented methods:

- None / sequential fine-tuning
- Joint Training
- EWC
- LwF
- A-GEM
- Separate Networks
- Generative Classifier
- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW

The older GMvandeVen-based scripts are kept only as historical reference files under [`code/legacy_reference/`](code/legacy_reference/).

## Main Results And Graphs

Most important comparison graph:

![Paper vs GMvandeVen vs from-scratch](assets/paper_vs_gmvandeven_vs_from_scratch.png)

Other graphs:

- [`assets/all-methods-by-scenario.png`](assets/all-methods-by-scenario.png)
- [`assets/accuracy-heatmap.png`](assets/accuracy-heatmap.png)
- [`assets/lsr-ablation-by-scenario.png`](assets/lsr-ablation-by-scenario.png)
- [`assets/selected-learning-curves.png`](assets/selected-learning-curves.png)
- [`assets/from_scratch_classic_no_lsr_2000_final_accuracy.png`](assets/from_scratch_classic_no_lsr_2000_final_accuracy.png)
- [`assets/from_scratch_classic_no_lsr_2000_learning_curves.png`](assets/from_scratch_classic_no_lsr_2000_learning_curves.png)

Result files:

- [`assets/paper_vs_gmvandeven_vs_from_scratch.csv`](assets/paper_vs_gmvandeven_vs_from_scratch.csv)
- [`assets/from_scratch_classic_no_lsr_2000_summary.csv`](assets/from_scratch_classic_no_lsr_2000_summary.csv)
- [`assets/splitMNIST_2000_all_scenarios_summary.csv`](assets/splitMNIST_2000_all_scenarios_summary.csv)

## Video Helper

Use [`lsr_animation.html`](lsr_animation.html) as a short visual animation for explaining LSR-lite and its variants in the project video.

## Reports

- [`assets/code_explanation_full.docx`](assets/code_explanation_full.docx)
- [`assets/full_project_explanation_with_all_graphs.docx`](assets/full_project_explanation_with_all_graphs.docx)
- [`assets/summary_hebrew_splitMNIST_2000.docx`](assets/summary_hebrew_splitMNIST_2000.docx)

## Documentation Index

- [`docs/README.md`](docs/README.md)
- [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md)
- [`docs/REPRODUCTION_REPORT.md`](docs/REPRODUCTION_REPORT.md)
- [`docs/CODE_EXPLANATION.md`](docs/CODE_EXPLANATION.md)
- [`docs/METHODS_IMPLEMENTATION.md`](docs/METHODS_IMPLEMENTATION.md)
- [`docs/PAPER_COMPARISON.md`](docs/PAPER_COMPARISON.md)
- [`docs/IMPLEMENTATION_AUDIT.md`](docs/IMPLEMENTATION_AUDIT.md)
- [`docs/LSR_LITE_EXPLANATION.md`](docs/LSR_LITE_EXPLANATION.md)

## GitHub Pages Setup

Use:

```text
Branch: main
Folder: / root
```

The repository uses root [`index.html`](index.html), so do not choose `/docs` for GitHub Pages.
