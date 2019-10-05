# Accelerating Deep Learning by Focusing on the Biggest Losers

This paper introduces Selective-Backprop, a technique that accelerates the
training of deep neural networks (DNNs) by prioritizing examples with high loss
at each iteration. Selective-Backprop uses the output of a training example's
forward pass to decide whether to use that example to compute gradients and
update parameters, or to skip immediately to the next example. By reducing the
number of computationally-expensive backpropagation steps performed,
Selective-Backprop accelerates training. Evaluation on CIFAR10, CIFAR100, and
SVHN, across a variety of modern image models, shows that Selective-Backprop
converges to target error rates up to 3.5x faster than with standard SGD and
between 1.02--1.8x faster than a state-of-the-art importance sampling approach.
Further acceleration of 26% can be achieved by using stale forward pass results
for selection, thus also skipping forward passes of low priority examples.

## Paper
*Angela H. Jiang, Daniel L.-K. Wong, Giulio Zhou, David G. Andersen, Jeffrey Dean, Gregory R. Ganger, Gauri Joshi, Michael Kaminksy, Michael Kozuch, Zachary C. Lipton, Padmanabhan Pillai*

## Code

Selective-Backprop is integrated into two existing repositories:
1. [`pytorch-cifar10`](https://github.com/kuangliu/pytorch-cifar): We build Selective-Backprop using this as a base.
2. [`Cutout`](https://github.com/uoguelph-mlrg/Cutout): We run experiments using this implementation of Cutout for CIFAR10, CIFAR100 and SVHN.
