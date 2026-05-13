# Code Included In This Report Repository

## Final Submission Code

Use this folder:

`from_scratch/`

It contains the clean-room implementation required by the assignment.
It does not import the GMvandeVen repository.

Main files:

| File | Purpose |
|---|---|
| `from_scratch/splitmnist_cl.py` | Independent Split MNIST continual-learning runner. |
| `from_scratch/run_all_classic_from_scratch.ps1` | Runs all classic methods over Class-CL, Domain-CL, and Task-CL without rerunning LSR. |
| `from_scratch/run_splitmnist_from_scratch.ps1` | Runs long experiments for Class-CL, Domain-CL, or Task-CL. |
| `from_scratch/aggregate_from_scratch.py` | Creates combined CSV files and graphs. |
| `from_scratch/compare_reproduction_sources.py` | Compares paper results, GMvandeVen-code runs, and our from-scratch runs. |
| `from_scratch/plot_from_scratch_summary.py` | Creates a final-accuracy graph from `summary.csv`. |
| `from_scratch/README.md` | Commands, method list, and protocol notes. |

Implemented from scratch:

- None
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

## Legacy Reference Files

The other files in this folder come from the earlier phase of the project, when
we used the GMvandeVen repository as a runnable reference implementation.

They are kept for transparency, but they are not the final answer to the
assignment requirement that every method be implemented by us.

| File | Purpose |
|---|---|
| `train_lsr_lite.py` | Earlier LSR-lite prototype that reused repository utilities. |
| `run_phase1_splitmnist_class_2000.ps1` | Earlier Class-CL runner for the GMvandeVen repo. |
| `run_phase2_splitmnist_domain_2000.ps1` | Earlier Domain-CL runner for the GMvandeVen repo. |
| `run_phase3_splitmnist_task_2000.ps1` | Earlier Task-CL runner for the GMvandeVen repo. |
| `phase1_summarize.py` | Earlier Class-CL summary and graph generator. |
| `phase2_summarize.py` | Earlier Domain-CL summary and graph generator. |
| `phase3_summarize.py` | Earlier Task-CL summary and graph generator. |
| `patches/eval_history_and_options.patch` | Earlier patch for learning-curve logging in the GMvandeVen repo. |

## Protocol Notes For The New Code

- Test data is used only for evaluation.
- Replay buffers are built only from train data.
- Class-CL evaluates over all 10 classes without task identity.
- Task-CL uses task identity through allowed-class masking.
- A-GEM and LSR variants use the same default memory budget: 100 samples per original digit class.
