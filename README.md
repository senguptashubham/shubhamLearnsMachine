# shubhamLearnsMachine

A learning memoir of my AI ML journey — small scripts and projects as I work through concepts, not a polished library. Each subfolder is a self-contained topic.

## Contents

| Folder | What's in it |
|---|---|
| [`pytorch-basics/`](pytorch-basics/) | Small scripts covering PyTorch tensor fundamentals — creation, numpy interop, device placement, indexing/broadcasting, and autograd. |
| [`mnist-numpy-ann/`](mnist-numpy-ann/) | A neural network built from scratch in raw NumPy (forward pass, backprop, gradient descent — no autograd frameworks) trained on MNIST digit classification. Includes the debugging process and training results. |

## Setup

Environment used: a conda env (`mlwork`) with `numpy`, `matplotlib`, `torch`, and `keras` (Keras is only used for MNIST data loading / one-hot encoding, not for building or training any model here).
