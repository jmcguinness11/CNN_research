This directory contains four convolutional neural networks that work on the CIFAR-10 
dataset.  Two of these do not have a fully-connected layer, while the other two do.
One of the FCL networks, [train_upperrelu_cifar.py](./train_upperrelu_cifar.py), 
contains four convolutional layers and then a fully-connected layer, and 
[the other](./train_alexnet_baseline_cifar.py) is the famous [AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
developed by Alex Krizhevsky in 2012. Of the no-FCL networks, one uses our normal 
approach to calculating loss and determining correct predictions without a FC layer 
(see [cnn_no_fcl](../../cnn_no_fcl) for details), and the other just uses an average 
of the max and min points in each of the ten maps output by the final convolutional 
layer.  These two networks were explored further with MNIST, so more related 
information can be found in the [ten_class_mnist folder](../ten_class_mnist).
