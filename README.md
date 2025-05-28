# 🧠 Deep Learning From Scratch — CNN, RNN, Transformer, Knowledge Distillation

A hands-on implementation of core deep learning architectures using only low-level code in **C++ and Python**, without relying on high-level frameworks like PyTorch or TensorFlow.

This project is meant to **demystify the internals** of deep learning by building models from the ground up — including **forward and backward passes**, and training loops.

## 🔧 Implemented Architectures

### 🌀 Convolutional Neural Network (CNN)

* Implemented from scratch in **both C++ and Python**
* Includes:
    * Convolutional layers 
    * Max pooling
    * Fully connected layers
    * Manual backpropagation for all components


### 🔁 Recurrent Neural Network (RNN)

* Implemented in **Python**
* Features:
    * Step-by-step forward and backward propagation through time (BPTT)
    * Trained on synthetic sequential data

### 🔺 Transformer

* Implemented in **C++ and Python** following the original paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
* Includes:
    * Scaled dot-product attention
    * Multi-head attention
    * Positional encoding
    * Feedforward layers
    * Manual implementation of layer normalisation and residual connections
    * C++ version mirrors Python implementation for performance


### 🧪 Knowledge Distillation

* Trained a large CNN on an image classification task
* Used knowledge distillation to transfer its learning to a smaller student network
* Based on [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)
* Implemented training pipelines and soft-label transfer manually

## 🚀 Why This Project Matters

This repository demonstrates:

* Mastery of the **core principles** of deep learning
* Ability to implement complex systems **from first principles**
* Proficiency in both **high-level (Python)** and **low-level (C++)** languages
* Clear understanding of **gradient flow**, **optimisation**, and **architecture-specific quirks**