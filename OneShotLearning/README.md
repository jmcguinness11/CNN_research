## One-Shot Learning
One of the major problems with modern machine learning is that networks tend to need
to be presented with a massive amount of data in order to effectively learn.  Here
we explore the One-Shot Learning framework that attempts to solve this problem, 
an idea pioneered by Google DeepMind.  Our networks are based off 
<a href="https://arxiv.org/abs/1606.04080">this paper</a> from DeepMind published in 
June 2016.

Our general structure is TODO TDASFKLAHDFLAKFDLAHF

One-Shot vs. Zero-Shot Learning
-------------------------------
WHAT YOU TEST ON

The Datasets
------------
The two datasets we use here are MNIST and the 
<a href="https://github.com/brendenlake/omniglot">
Omniglot dataset for one-shot learning</a>. The omniglot dataset consists of 1623
different types of handwritten characters (20 examples per character) from 50 
different alphabets.  In our code, we select five of these classes at random, and 
each time through the network we generate a support set consisting of two examples
from each class, and one additional example from the query class, and run all these
through the network as described above.
