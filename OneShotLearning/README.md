## One-Shot Learning
One of the major problems with modern machine learning is that networks tend to need
to be presented with a massive amount of data in order to effectively learn.  Here
we explore the [One-Shot Learning](https://en.wikipedia.org/wiki/One-shot_learning)
framework that attempts to solve this problem. Our networks are based off 
<a href="https://arxiv.org/abs/1606.04080">this paper</a> from Google DeepMind 
published in June 2016.

## The Datasets
The two datasets we use here are MNIST and the 
<a href="https://github.com/brendenlake/omniglot">
Omniglot dataset for one-shot learning</a>. The omniglot dataset consists of 1623
different types of handwritten characters (20 examples per character) from 50 
different alphabets.  In our code, we select five of these classes at random, and 
each time through the network we generate a support set consisting of two examples
from each class, and one additional example from the query class, and run all these
through the network as described below.  To use MNIST in a way that makes sense for
One-Shot learning, I had to shorten the train data, just taking ten elements from
each class.  Using the whole dataset would defeat the purpose of One-Shot learning,
which tries to learn from a limited amount of data.

## Network Structure
Our One-Shot networks use a three-layered convolutional neural network with no
fully-connected layer that simply outputs feature maps.  We select a query class
and run one element from this class through the network.  Then we pick two support
elements from each class (including the query class) and run each of these through
the same network *(technically these are actually separate networks, but we code them
to share weights, so in practice they are the same network)*.  We then calculate the
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between each
of the supports and the query, combining the similarities of supports from the same
class by averaging them.  The class of the group of support elements that are the 
most similar to the query element is the guess for the correct class.  A softmax
cross entropy loss function is used to try to train the network to guess correctly
more often.

## One-Shot vs. Zero-Shot Learning
However ambitious the One-Shot learning's aim to learn classes from a limited amount
of input data is, Zero-Shot learning is even more bold.  Instead of trying to learn
specific classes, a Zero-Shot network is simply trained to separate elements from
different classes as much as possible.  The aim is to successfully use such a network
on elements from classes it has never seen before.  In our implementation using
Omniglot, for instance, we use 50 classes for training and 50 entirely different
classes for testing, trying to correctly select from 5 classes at a time.

## Results
|Network  |# Classes  |Dataset  |Accuracy     |
|One-Shot-|10---------|MNIST----|about 85-90%-|
|One-Shot-|5----------|Omniglot-|about 85-100%|
|Zero-Shot|5----------|Omniglot-|about 55-75%-|
*Note: Because of the nature of One-Shot learning, obtaining more specific accuracy
measurements is not possible.*

