# Dynamic Convolutional Neural Networks

### Introduction
This is a Theano implementation of the paper "A Convolutional Neural Network for Modelling Sentences" (<a href="http://nal.co/papers/Kalchbrenner_DCNN_ACL14">click here</a>.
The example included is that of binary movie review sentiment classification (Stanford Sentiment Treebank).
I was able to achieve a test set accuracy of 85-86% which is just below the reported accuracy of 86.8%.

### Paper issues
Not all training details are clear from the paper. 
A Matlab implementation is provided by the authors but that implementation is different from the paper.
E.g., different number of layers and filters.
Some training details are also different (or not reported) in the paper. 
For example, the L2 regularization is very detailed (different values for different matrices).
Adagrad was used to train the network but according to the code they reset the accumulated gradient.

### Implementation details
The layers of the network are wrapped as Lasagne layers and can be easily reused.
In the paper, some layer types were introduced which are not trivial for Theano.
(1) 1D convolution layers that only apply row wise convolutions and not on all rows at once. 
(2) Dynamic K-max pooling. Currently an argsort operation is used which is executed on the CPU. 
However, this operation is too heavy for selecting the K max values.

Because of these implementations, a very heavy GpuContiguous operation is automatically introduced somewhere.
If you have a solution or comments, I'm happy to support pull request ;)
