# Project Scope

## Chosen Paper

The project is based on:

van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). **Three types of incremental learning**. *Nature Machine Intelligence*.

Source: https://www.nature.com/articles/s42256-022-00568-3

Original reference repository:

https://github.com/GMvandeVen/continual-learning

## Why This Paper Was Chosen

The paper is a good fit for a reproduction project because it studies catastrophic forgetting directly and separates continual learning into three clear scenarios:

- Class-Incremental Learning (Class-CL)
- Domain-Incremental Learning (Domain-CL)
- Task-Incremental Learning (Task-CL)

It also reports results on Split MNIST, which is small enough to run locally on an RTX 3070 while still showing the main continual-learning failure modes.

## What We Reproduced

We focused on the Split MNIST results from the paper and reproduced the main methods for:

- Class-CL
- Domain-CL
- Task-CL

The main comparison methods were:

- None / sequential fine-tuning
- Joint Training
- EWC
- LwF
- A-GEM
- Separate Networks for Task-CL
- Generative Classifier as a simplified clean-room variant

We also added our own experimental prototype:

- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW

## What Is Not A Full Reproduction

This repository does not reproduce every dataset, figure, and method in the full paper. It focuses on Split MNIST and the methods we could implement and run locally.

Known limitations:

- The paper reports mean and SEM over 20 seeds; most of our long runs are single local runs.
- Our clean-room EWC implementation is valid but does not fully match the original Task-CL EWC result.
- Our clean-room Generative Classifier is a simple diagonal Gaussian classifier, not necessarily the same as the strongest generative setup in the paper.
- LSR-lite is our own experimental method, not a method from the paper.

## What To Present As The Main Implementation

Use:

`code/from_scratch/`

This folder contains the assignment-compliant implementation written for this project.

The earlier GMvandeVen-based runs are kept only as reference results and for transparency.
