This directory contains four convolutional neural networks that work on the CIFAR-10 
dataset.  Two of these do not have a fully-connected layer, while the other two do.
One of the FCL networks, *train_upperrelu_cifar.py*, contains four convolutional 
layers and then a fully-connected layer, and the other is the famous AlexNet
developed by Alex Krizhevsky in 2012. Of the no-FCL networks, one uses our normal 
approach to calculating loss and determining correct predictions (see the 
*cnn_no_fcl* directory for details), and the other just uses an average of the max 
and min points in each of the ten output maps.  These two networks were explored
further with MNIST, so more related information can be found in the
*SpecificDatasetWork/ten_class_mnist/* folder.

FUTURE WORK USING JUST AVERAGE OF ENTIRE OUTPUT MAP (VERY CeNN-friendly)
