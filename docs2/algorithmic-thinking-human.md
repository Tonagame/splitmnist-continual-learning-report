# Planned Implementation Steps
Steps as seen in the project topic defense:
1) Set up the Python environment:
	- download relevant python version
	- necessary libraries
	- datasets 
2) Download and run the code from the Github repository to create a reference point to compare the results
3) Implement the baseline methods and train the models on the MNIST dataset:
	- "None" - train the model on the tasks in the standard way (no catastrophic forgetting mitigation)
	- "Joint" - the model was trained on all of the data at the same time (no splitting to "tasks")
4) Research each method to get an understanding of implementation difficultly (omit methods beyond the scope of the project)
5) Implement the methods in the following order: 
	- Separate Networks
	- EWC
	- LwF
	- A-GEM
	- Generative classifier
6) Train each model (for each type of CL and method) on the MNIST dataset, matching the hyperparameter to the ones from the article.
   Initially setting a constant seed to guarantee the same data split groups between iteration.
7) After verifying the model works, a more random seed can be used to get slightly different results and average them. 
8) Compile the results in a table\graph and compare the performance of each method
9) Compare our result to the ones in the article and check for discrepancies between them (fix any problems that arise)
10) Repeat steps 6-9 on the  CIFAR-100 dataset
11) (possible expansion) change the hyper-parameters and check the impact on the results

# Implementation Steps taken
Documentation of the different components and stages of the project
## Environment Setup
Created a local python environment using Anaconda
- download relevant python version
- necessary libraries (including GPU acceleration libraries)
- MNIST dataset

Verified everything was installed correctly

## Reference Repository Setup and Running
Cloned the Github repo from the paper and checked everything works by running a small 'Smoke test' on the following scenarios:
- Separate Networks
- EWC
- LwF
- A-GEM

The 'Smoke Test' is a light run with small parameter sizes, meant to quickly catch errors, check if the code runs on the GPU and allow iterative testing without the time consuming wait on larger iterations. (The test was performed on our methods as well)

After successfully running the initial test, the full scenarios were performed, and the results were compared against the paper's to check the reference ran well.
## Core Functions
A python script with the core functions utilized by the different methods/scenarios, containing function for:
- dataset loading and splitting into contexts
- Creating the ML model
- Move data to/from the GPU
- Saving the results to CSV and JSON files to later aggregate and create graphs

## Command Line Interface
A command line interface was built to easily choose what scenario + method to run and dictate the running parameters (number of iterations per context, batch size,  seed, etc.)

The program parses the input from the command line, runs the selected CL scenario and method to run and saves the run metrics.

## Baseline Methods 
Like in the paper, to create a lower and upper bound for the results the Joint and None methods were implemented:
**Joint** - training the model on the whole dataset without dividing into contexts - giving an ideal result where the model learns all the tasks without forgetting (Not continual learning)
A function to train the model on all datasets was implemented

**None** - The model was incrementally trained on the contexts without any CL method - it is expected to perform badly as it demonstrates catastrophic forgetting

A general function was written to train the model continuously on the tasks one after the other, getting the current CL method as an input parameter (with "None" being an option - any CL logic is omitted)

To check our work, the results were compared against the paper's and the locally run code from the paper - The results were very close.
## CL Methods
Starting with EWC, each method had its own python file created and filled with functions specific to it:
- EWC - the fisher diagonal estimation and EWC penalty (Loss) calculation
- Separate Networks - training a sub-model for each context
- LwF - creating a snapshot of the model to be used as a teacher, performing knowledge distillation and calculating the loss
- A-GEM - calculating the vector gradient of the current context and of the replay, and projecting the current gradient if it strays too far from the replay's

After implementing each method, it was ran and compared against the reference results. Some methods were closer to the paper than others - with most being really close and a few were off by a margin.

## Hybrid approach
Taking inspiration from a method called Latent Replay, we implemented a CL method fusing A-GEM and LwF: taking the replay element from A-GEM and the knowledge distillation from LwF.

The goal is to reduce catastrophic forgetting by keeping a small, fair memory of old training examples and preserving both the model's old output behavior and its internal feature representation.
For each saved replay example, the method stores:
- image `x`
- label `y`
- teacher logits at insertion time
- penultimate feature vector at insertion time

The loss function:
```
L_total =
  L_CE_current
+ L_CE_replay
+ lambda_kd   * L_KD_logits
+ lambda_feat * L_feature_anchor
```

Where:
- `L_CE_current` learns the current context.
- `L_CE_replay` relearns labels of old replay examples.
- `L_KD_logits` keeps the current model close to the old teacher predictions.
- `L_feature_anchor` keeps internal feature vectors close to their stored values.

The method displayed a strong improvement over the others - scoring high accuracy in all CL scenarios.
### Additions to the method
We experimented with small additions to the method to check if they have any significant impact on the result. both additions were used separately and in combination - yielding small gains but no real improvement over just using the new method.
#### Fourier
The Fourier transform (FFT) of the feature vectors was used a regularization component in the loss calculation.
#### Adaptive Stability Weighting (AWS)
The ratio between the loss of the replay and the current loss was used as a component of the knowledge distillation loss, making it more/less significant in the loss calculation.

## Validation during Development
Important checks:
- Test data is not used for training.
- Replay memory is built only from train data.
- Class-CL evaluation does not use task identity.
- Class-CL final evaluation uses all 10 classes.
- Task-CL uses allowed-class masking, as required by the protocol.
- A-GEM and LSR-lite use the same default memory budget: 100 samples per original digit class.

For large, 2000-iteration (per context) runs, checked:
- every method produced logs,
- failed methods did not stop the whole experiment,
- `run_status.csv` recorded success/failure,
- final accuracy was written,
- learning curves were written,
- graphs were generated after aggregation.

## Compiling the data
After all the CL methods and scenario combinations were tested and checked - graphs were made making different comparisons between the methods:

- Our results compared to the paper's results and the locally ran code from the included repository
- The performance of each method grouped by scenario
- The learning curves of each method
