# Project Video

The assignment asks for a short video, about 2-5 minutes.

## Video Link

Replace this placeholder after recording:

`TODO: paste YouTube / Google Drive / other video link here`

## Suggested Video Structure

1. Introduce the project: Split MNIST continual learning.
2. Explain the problem: catastrophic forgetting.
3. Explain the three scenarios: Class-CL, Domain-CL, Task-CL.
4. Show the website and graphs.
5. Explain the main result: LSR-lite works especially well in Class-CL.
6. Mention that Fourier and ASW were ablations, not the core mechanism.
7. End with the final conclusion.

## Short Script

This project tested continual learning on Split MNIST.
The model learns several contexts one after another, and the main challenge is that it can forget older contexts after learning new ones.

I compared several methods, including EWC, LwF, A-GEM, Separate Networks, Joint Training, and an experimental method called LSR-lite.
LSR-lite stores real replay samples from old training data together with labels, teacher logits, and feature vectors.

The hardest setting was Class-CL, because the model did not receive task identity during evaluation.
In that setting, the None baseline collapsed, while LSR-lite reached much higher accuracy.

The final conclusion is that real replay with distillation and feature anchoring is a strong direction for continual learning.
Fourier and ASW were useful experiments, but the main reason the method worked was the real replay memory and stored teacher signals.
