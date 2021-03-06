In this directory, I investigated the effects of using a new nonlinearity with
the potential to speed up neural network training process.  As the nonlinearity
is essentially RELU with an upper bound, I call it **"Upper ReLU"**.

What is Upper ReLU?
-----------------------
Upper ReLU is a nonlinearity that tries to squeeze values between -1 and 1.

The basic form of Upper ReLU is simple and exactly matches the standard nonlinearity
for [Cellular Neural Networks](http://www.scholarpedia.org/article/Cellular_neural_network).
<img src="http://www.scholarpedia.org/w/images/c/c3/CNN_output.png" alt="CeNN Nonlinearity" width="180">
```
(∞, -1): y = -1 + ⍺(x+1)
[-1, 1]: y = x
(1, ∞):	 y = 1 - ⍺(x-1)
```

However, András and I decided to make Upper ReLU "leaky" to ensure values don't get 
stuck at the endpoints of -1 and 1.  This regime forces values back into the [-1,1] 
range.

<img src="./UpperRelu.png" alt="Upper ReLU" width="180">

```
(∞, -1): y = -1 + ⍺(x+1)
[-1, 1]: y = x
(1, ∞):	 y = 1 - ⍺(x-1)
```
This can easily be implemented in TensorFlow as follows:
```python
alpha = .01
ReLU = tf.maximum(-1+alpha*(Input+1), Input)
ReLU = tf.minimum(1-alpha*(ReLU-1), ReLU)
```


How did I find this nonlinearity?
-------------------------------------
When working on removing the fully-connected layer (FCL) from our convolutional 
neural networks (CNNs), I noticed an interesting problem with our approach.  Lets 
say we are working with ten classes.  Essentially, instead of a FCL, we compare our 
ten output maps (one for each of the ten classes) from the last convolutional layer 
to ten target maps, 9 of all -1s and 1 filled with 1s in the position corresponding 
to the correct label. See [here](../cnn_no_fcl) for more information on our 
approach. I "discovered" the upper ReLU nonlinearity when trying to speed up the slow
convergence time associated with this approach. Since ReLU is only bounded on one 
end, the distances to the 1 maps were consistently greater than the distances to the 
-1 maps until the network was trained enough to standardize the differences.  
Until these differences were standardized, the network would perform terribly.  

Once you understand this issue, it is somewhat obvious why Upper ReLU would result
in a significant training speedup.  However, it is not as clear that it would work
similarly in other network architectures.  András decided to test Upper ReLU on a 
simple CNN with a FCL as well as [AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf) and noticed a similar speedup, so we 
then decided to delve into an in-depth study of the nonlinearity.

Initial Findings
----------------
After I figured out its potential, I ran some quick initial tests on Upper ReLU.
Those results are stored in the [InitialFindings](./InitialFindings) folder. 

What We Tested
--------------
We were confident that Upper ReLU speeds up training time compared to normal or
leaky ReLU, but before going forward too far, we did a literature search to find
other approaches researchers have successfully used to speed up training time.
The most relevant methods we found were 
[batch normalization](https://arxiv.org/abs/1502.03167) and the exponential linear
unit ([ELU](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf)).
Prenormalizing the input data by squeezing it all into the ReLU scheme ([0,1] for 
normal ReLU and [-1,1] for Upper ReLU) before the network starts learning can also
help. I ran tests on both MNIST and CIFAR-10 for networks with and without 
batch-normalization, pre-normalization, Upper ReLU, ELU, and some combinations 
thereof, storing the results as numpy data files in the [results](./results) folder.

Some Interesting Results
------------------------
In all of the following graphs, the network with Upper ReLU is represented in green.
Training accuracy results are displayed on the right, while independent test
accuracies are shown on the left.

#### ReLU vs. Upper ReLU - MNIST
<img src="./relu_vs_upperrelu.png" alt="ReLU vs. Upper ReLU" width="700">

#### Batch Normalization (normal ReLU) vs. Upper ReLU - CIFAR
<img src="./bnorm_vs_upperrelu_cifar.png" alt="B-Norm vs. Upper ReLU" width="700">

*This result is particularly interesting.  It suggests that while batch 
normalization allows a network to learn faster, it badly hurts its ability to learn
the data.  This is a strange result because the literature seems to show that batch
normalization is extremely effective, so we would certainly welcome feedback.
However, we have thoroughly checked our implementation and can find nothing wrong, so
we believe that with our CNN, batch normalization helps increase train accuracy while
badly hurting independent test accuracy (a horrible case of overfitting).  Adding
dropout did not help.*

Using the Graphing File:
------------------------
To explore these results for yourself, you can use the 
[graph_aggregates.py](./graph_aggregates.py) script.  This script depends on
**matplotlib** and **numpy**, so if you do not have those installed, you will have
to do so as follows:
```
pip install matplotlib
pip install numpy
```
Now that you have the dependencies to run the file, you can generate graphs by
running the following command:
```
python graph_aggregates.py
```

To look at train data, uncomment the second lines and comment out the first two lines
under *cut off data to a standard length* (line 13), and to look at the test data, 
leave it as is (with the first two lines commented out.  The data file names that can
be examined can be found under [results](./results).  To examine these, change the
file names under *actual loading* (line 9) with a proper file name.

I apologize for the less than ideal state of this program.  If I have time in the
future, I will add command line options to the script to make it easier to use.
