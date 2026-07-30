# shubhamLearnsMachine

A learning memoir of my AI ML journey — small scripts and projects as I work through concepts, not a polished library. Each subfolder is a self-contained topic.

## Contents

| Folder | What's in it |
|---|---|
| [`pytorch-basics/`](pytorch-basics/) | Small scripts covering PyTorch tensor fundamentals — creation, numpy interop, device placement, indexing/broadcasting, and autograd. |
| [`mnist_pytorch_mlp/`](mnist_pytorch_mlp/) | A custom MLP (`nn.Module`) trained on MNIST, built step-by-step: manual training loop → reusable training function → comparing Full-batch vs Mini-batch vs Stochastic Gradient Descent on convergence speed and overfitting risk. |
| [`mnist-numpy-ann/`](mnist-numpy-ann/) | A neural network built from scratch in raw NumPy (forward pass, backprop, gradient descent — no autograd frameworks) trained on MNIST digit classification. Includes the debugging process and training results. |
| [`cnn/my_conv2d/`](cnn/my_conv2d/) | A from-scratch Conv2D layer (`MyConv2d`, im2col + matmul) verified for correctness/gradients against `nn.Conv2d` and benchmarked CPU vs GPU across problem sizes. See [cnn/my_conv2d/README.md](cnn/my_conv2d/README.md) for the full write-up and findings. |
| [`cnn/cifar10/`](cnn/cifar10/) | A CNN trained on CIFAR-10, improved step-by-step from 73.86% → 82.43% test accuracy purely through training methodology — data augmentation, batch norm, best-checkpoint restoration, patience-based early stopping, and a random hyperparameter search — with confusion-matrix/misclassified-image analysis at the end. See [cnn/cifar10/README.md](cnn/cifar10/README.md) for the full timeline and findings. |

## Setup

Environment used: a conda env (`mlwork`) with `numpy`, `matplotlib`, `torch`, and `keras` (Keras is only used for MNIST data loading / one-hot encoding, not for building or training any model here).
