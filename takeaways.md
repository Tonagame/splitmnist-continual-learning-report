# Takeaways: Split MNIST Continual Learning Project

This project explored continual learning on Split MNIST using the GMvandeVen `continual-learning` repository.
The goal was to understand how different methods handle catastrophic forgetting when a neural network learns a sequence of contexts instead of receiving all data at once.

## What I Built

I set up the repository locally on a Windows PC with an NVIDIA RTX 3070 GPU.
After confirming that Python, Conda, Git, CUDA, and PyTorch were working, I ran Split MNIST experiments across three continual-learning scenarios:

- Class-CL
- Domain-CL
- Task-CL

I compared standard methods such as None, EWC, LwF, A-GEM, Separate Networks, Generative Classifier where supported, and Joint Training.
I also implemented and tested an experimental prototype called LSR-lite.

LSR-lite stores real replay examples from previous training data.
For each stored example it keeps:

- the image
- the label
- teacher logits at insertion time
- a penultimate feature vector at insertion time

During training, LSR-lite combines cross entropy, replay, logit distillation, and feature anchoring.
I also tested two ablations:

- Fourier auxiliary regularization
- Adaptive Stability Weighting

## Algorithmic Thinking

The main algorithmic challenge is the stability-plasticity tradeoff.
A model needs plasticity to learn new tasks, but it also needs stability so that old knowledge is not destroyed.

Each method handles this tradeoff differently.
EWC protects parameters that seem important for old tasks.
LwF tries to preserve the behavior of the previous model.
A-GEM uses replay examples to avoid harmful gradient updates.
Separate Networks avoids interference by using one network per task.
LSR-lite combines real replay samples with teacher logits and feature anchors, so it gives the model direct reminders of old data and old internal representations.

## What The Results Showed

The most important lesson was that the scenario matters a lot.

Class-CL was the hardest setting because the model did not receive task identity during evaluation.
In this setting, the None baseline collapsed to about 0.198 accuracy.
A-GEM helped, but LSR-lite was much stronger and reached above 0.92 accuracy.
This showed that real replay plus distillation and feature anchoring is very effective for class-incremental learning.

Domain-CL was easier than Class-CL.
Plain LSR-lite was the best non-Joint method in this scenario.
Fourier and ASW did not improve it.

Task-CL was the easiest setting because the evaluation protocol gives task identity through allowed classes.
In this case, LwF and Separate Networks almost matched Joint Training.
LSR-lite still performed well, but it was not the best method for Task-CL.

## What I Learned About LSR-lite

LSR-lite is most promising when task identity is not available.
It worked especially well in Class-CL, which is the most realistic and difficult scenario in this project.

The core mechanism was not Fourier or ASW.
The useful part was:

- storing real old examples
- storing teacher logits
- storing feature vectors
- replaying old examples during new training
- anchoring the model's internal representation

Fourier was a useful ablation and gave a small improvement in Class-CL, but it did not consistently help.
ASW was stable, but it also did not consistently improve performance.

## Testing And Validation

The project used smoke tests before serious runs.
The final serious experiments used:

- 5 contexts
- 2000 iterations
- batch size 128
- acc-n 1024
- full-test final evaluation

I also checked important protocol details:

- test data was not used during training
- replay buffers were built from train data only
- A-GEM and LSR-lite used the same buffer budget
- Class-CL did not use task identity
- Task-CL kept the repository's original allowed-classes evaluation

## Final Reflection

This project helped me understand that continual learning is not just about making a model accurate on the next task.
The real challenge is preserving useful old knowledge while still learning new information.
The results made the differences between Class-CL, Domain-CL, and Task-CL very clear.

My strongest conclusion is that replay methods with meaningful stored signals are very powerful.
LSR-lite is not only storing old examples; it is also storing what the model believed and how the model internally represented those examples.
That combination made it much stronger than several classic baselines in the hardest setting.

If I continued this project, I would test LSR-lite on harder datasets such as CIFAR-100 or TinyImageNet and compare different buffer sizes.
I would also tune the Fourier and ASW terms more carefully, because the current results suggest they are interesting but not yet reliably useful.
