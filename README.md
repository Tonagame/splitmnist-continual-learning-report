# Split MNIST Continual Learning Reproduction

This repository is the final GitHub submission package for a project on catastrophic forgetting and continual learning with Split MNIST.

It includes:

- a GitHub Pages report,
- Markdown documentation,
- Python implementation code,
- graphs,
- CSV result files,
- reflective writing,
- AI usage documentation.

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
- H&T
- H&T + Fourier
- H&T + ASW
- H&T + Fourier + ASW

The older GMvandeVen-based scripts are kept only as historical reference files under [`code/legacy_reference/`](code/legacy_reference/).

## Three Main Result Graphs

### 1. First We Ran Their Code And Compared It To The Paper

At the beginning, we ran the original GMvandeVen continual-learning code locally and compared those results against the paper table.

![Paper vs original-code local run](assets/paper_vs_ours_splitMNIST_common_methods.png)

Source files:

- [`assets/paper_vs_ours_splitMNIST_common_methods.png`](assets/paper_vs_ours_splitMNIST_common_methods.png)
- [`assets/paper_vs_ours_splitMNIST_common_methods.csv`](assets/paper_vs_ours_splitMNIST_common_methods.csv)

### 2. Then We Ran Our Code And Compared Everything

After that, we ran our own implementation and compared it against both the paper and the previous local run of the original code.

![Paper vs original code vs our implementation](assets/paper_vs_gmvandeven_vs_from_scratch.png)

Source files:

- [`assets/paper_vs_gmvandeven_vs_from_scratch.png`](assets/paper_vs_gmvandeven_vs_from_scratch.png)
- [`assets/paper_vs_gmvandeven_vs_from_scratch.csv`](assets/paper_vs_gmvandeven_vs_from_scratch.csv)
- [`assets/from_scratch_classic_no_ht_2000_summary.csv`](assets/from_scratch_classic_no_ht_2000_summary.csv)

### 3. Finally We Compared H&T To The Other Methods

H&T is our experimental hybrid method: A-GEM-style replay memory plus LwF-style distillation, with feature anchoring. We also tested H&T + Fourier, H&T + ASW, and H&T + Fourier + ASW.

![H&T compared to the other methods](assets/all-methods-by-scenario.png)

Source files:

- [`assets/all-methods-by-scenario.png`](assets/all-methods-by-scenario.png)
- [`assets/h-and-t-ablation-by-scenario.png`](assets/h-and-t-ablation-by-scenario.png)
- [`assets/splitMNIST_2000_all_scenarios_summary.csv`](assets/splitMNIST_2000_all_scenarios_summary.csv)

All graph previews are collected in [`assets/README.md`](assets/README.md).

## Documentation Index

- [`doc/README.md`](doc/README.md) - updated final submission pack. This is the cleaner final version of the original documentation archive.
- [`doc/algorithmic-thinking-human.md`](doc/algorithmic-thinking-human.md)
- [`doc/ai-documentation.md`](doc/ai-documentation.md)
- [`doc/results-and-comparison.md`](doc/results-and-comparison.md)
- [`doc/haim-takeaway.md`](doc/haim-takeaway.md)
- [`doc/takeaways_Tamir.md`](doc/takeaways_Tamir.md)
