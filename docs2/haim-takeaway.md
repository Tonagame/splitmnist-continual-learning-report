# Haim Takeaway

For our Python course final project, we were asked to reproduce results from a scientific paper about catastrophic forgetting in machine learning. Catastrophic forgetting happens when a model learns a new task and then forgets tasks it learned before. The paper we chose compared several continual learning methods across three scenarios: Class-CL, Domain-CL, and Task-CL.

The first stage of our project was running the original code from the paper. This gave us a reference point. We ran the methods locally and compared the results to the paper's table. The results were mostly close, but not always identical, because the paper used multiple runs and averaged them, while we had more limited hardware and time.

The second stage was implementing our own version of the selected methods. We implemented methods such as None, Joint, EWC, LwF, A-GEM, and Separate Networks, then ran them on the same Split MNIST scenarios. After that, we compared three things: the paper results, the original code run, and our own code. This helped us check whether our implementation was reasonable and where it differed.

The third stage was trying our own method, H&T. H&T is a hybrid approach that combines ideas from LwF and A-GEM, with an additional feature anchoring mechanism. From A-GEM, it uses the idea of keeping a replay memory of old examples. From LwF, it uses knowledge distillation to preserve the old model's output behavior. Feature anchoring adds another stability signal by trying to keep the model's internal representation similar for old examples.

We also tested extra variations of H&T, including Fourier regularization and Adaptive Stability Weighting. These additions were meant to check whether extra stability mechanisms could improve the method. In practice, the core hybrid idea was the most important part, while the extra additions gave smaller changes.

Overall, this project taught me that reproducing machine learning results is not just about running code. It requires understanding the experiment, the dataset, the evaluation protocol, and the fairness of the comparison. It also taught me how to work with AI tools like Codex: the AI helped generate and organize code, but we still had to guide it, check it, fix mistakes, and make the important decisions.
