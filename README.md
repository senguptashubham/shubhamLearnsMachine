# shubhamLearnsMachine

A learning memoir of my AI ML journey — small scripts and projects as I work through concepts, not a polished library. Each subfolder is a self-contained topic.

## Contents

| Folder | What's in it |
|---|---|
| [`pytorch-basics/`](pytorch-basics/) | Small scripts covering PyTorch tensor fundamentals — creation, numpy interop, device placement, indexing/broadcasting, and autograd. |
| [`mnist_pytorch_mlp/`](mnist_pytorch_mlp/) | A custom MLP (`nn.Module`) trained on MNIST, built step-by-step: manual training loop → reusable training function → comparing Full-batch vs Mini-batch vs Stochastic Gradient Descent on convergence speed and overfitting risk. |
| [`mnist-numpy-ann/`](mnist-numpy-ann/) | A neural network built from scratch in raw NumPy (forward pass, backprop, gradient descent — no autograd frameworks) trained on MNIST digit classification. Includes the debugging process and training results. |
| [`cnn/`](cnn/) | A from-scratch Conv2D layer (`MyConv2d`, im2col + matmul) verified for correctness/gradients against `nn.Conv2d` and benchmarked CPU vs GPU across problem sizes — plus a CIFAR-10 CNN (in progress). See [cnn/README.md](cnn/README.md) for the full write-up and findings. |

## Setup

Environment used: a conda env (`mlwork`) with `numpy`, `matplotlib`, `torch`, and `keras` (Keras is only used for MNIST data loading / one-hot encoding, not for building or training any model here).
