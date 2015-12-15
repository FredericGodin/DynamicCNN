# Dynamic Convolutional Neural Networks

### Introduction
This is a Theano implementation of the paper "A Convolutional Neural Network for Modelling Sentences" (<a href="http://nal.co/papers/Kalchbrenner_DCNN_ACL14">click here</a>).
The example included is that of binary movie review sentiment classification (Stanford Sentiment Treebank).
I was able to achieve a test set accuracy of 85-86% which is just below the reported accuracy of 86.8%.

### Using it
To run it, simply run trainDCNN.py.
If you are only interested in the layers such as Dynamic K-max pooling, or the 1D convolution, only use the DCNN package.


### Paper/implementation issues
There is some discrepancy between the paper and Matlab code provided. Therefore, it was difficult to rely on the Matlab code for details not provided in the paper. For example:
(1) different number of layers and filters.
(2) the L2 regularization is not specified in the paper but is very detailed in the code (different values for different matrices). It would be hard to guess those values.

### Implementation details
The layers of the network are wrapped as Lasagne layers and can be easily reused.
In the paper, some layer types were introduced which are not trivial for Theano.
(1) 1D convolution layers that only apply row wise convolutions and not on all rows at once. 
(2) Dynamic K-max pooling. Currently an argsort operation is used which is executed on the CPU. 
However, this operation is too heavy for selecting the K max values.

Because of these implementation issues, a very heavy GpuContiguous operation is automatically introduced somewhere.
If you have a solution or comments, I'm happy to support pull requests ;)
