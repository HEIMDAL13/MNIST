# MNIST

This is the software code used for the experiments in the thesis: The effect of spatial RNNs on neural
network feature maps.

The project is developed in Pytorch 0.4 with a modular design that allows flexibility for running different experiments.

Visdom and TensorboardX are used for the visualization of the gradient distribution, the weight histograms and the training loss-accuracy.

The software implements multiple seed averaging, data sub-sampling and independent learning rate for different modules.

More than twenty different models that can be configured from a wide range of parameters are implemented.
