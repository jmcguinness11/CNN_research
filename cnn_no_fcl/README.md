## Cellular Neural Networks (CeNNs)
The code in this directory focuses on removing the fully connected layer from
convolutional neural networks (CoNNs) in an effort to make them friendly to an 
implementation on cellular neural networks (CeNNs).  CeNNs are like regular neural
networks, but they contain only local connections.  More specifically, they are
structered as a MxN grid of units that can only communicate with their direct
neighbors (see below photo). This property makes hardware CeNN implementations
extremely efficient. See 
[here](http://www.scholarpedia.org/article/Cellular_neural_network) for a more in
depth explanation of CeNNs.

<img src="http://www.scholarpedia.org/w/images/6/64/CNN_2D.png" width="300">

## CoNNs and CeNNs
The work done by the student who came to Budapest the summer before we did focused
on [implementing a CoNN with CeNNs](https://www.date-conference.com/proceedings-archive/2017/html/7026.html).  We wanted to expand on this work by attempting to remove
the fully-connected layer (FCL) from a CoNN.  A FCL is not very compatible with
CeNNs because of how CeNNs are restricted to local connections.  In order to create
a CeNN-friendly CoNN, we first had to fix some parameters in our CoNN.  Specifically,
we set the convolutional and pooling kernel sizes to 3x3, which is the exact
structure of units that one cell in a CeNN can communicate with as shown in the photo
above.  We then moved on to removing the FCL.

## Removing the FCL
In a normal CoNN, after the final convolutional layer, a fully connected layer is
included in which every neuron in the previous layer is connected to the next layer.
Before this FCL, the convolutional output consists of a certain number of feature
maps.  To turn these feature maps into a prediction and calculate a loss without
the FCL, we fix the number of output kernels/feature maps from the final layer to
be equal to the number of classes.  We then compare these maps to identically sized
maps consisting of either all negative ones or all ones.  These are set up in a
one-hot-like encoding, so all the maps are negative ones except for one map of all
ones in the position of the correct label.  For instance, if we are using MNIST and
the label is 0, the first of the ten maps will be 1s and the rest will be -1s. The
loss function tries to reduce mean square difference between the output maps and the
target one-hot maps.  The prediction is determined by assembling a list of mean square
differences from the output maps to every possible one-hot map and finding the index
of the smallest difference.  Say, for example, that the smallest difference is
from the output maps to the one-hot maps with ones in the fourth map, then the
prediction is 3 (the fourth number).  These networks without a FCL perform much
faster than those with a FCL, but there is definitely somewhat of an accuracy
dropoff.

## These Files
Besides [simple_cenn_network.py](./simple_cenn_network.py), which is a CeNN-friendly
CNN with 5 layers, the files in this folder are variants of one of two versions of
[AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf).  Both versions slightly simplify AlexNet to use input images sized
128x128 instead of 227x227 so that it can run faster (and because using AlexNet on
MNIST (28x28 images) and CIFAR-10 (32x32 images) is definitely overkill). The
**alexnet\_simplified** versions still have the same structure of 5 convolutional
layers followe by three FCLs, while the **cenn\_friendly** versions replace the
FCLs with an additional convolutional layer with ten output maps (one for each
class). The loss/accuracy for the **cenn\_friendly** networks is calculated as
described above under **Removing the FCL**.  The convolutional and pooling kernel
sizes were also all changed to 3x3 in this network so it would become CeNN-friendly.

## Results
I have run tests on each of these files and have the results stored in a private
Google sheet.  Please inquire if interested.
