All of the code in this directory works on the full ten class MNIST dataset.
The directory are three main categories of networks: [AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf), medium-complexity
convolutional neural networks (CNNs) with a fully-connected layer, and CNNs with no
fully connected layer (FCL).

*** Medium-complexity CNNs (fc\*)
These three networks only differ in the nonlinearity they use. The
[fc_benchmark_correct](./fc_benchmark_correct.py) network uses
[leaky ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs),
the [fc_upperrelu](./fc_upperrelu.py) network uses [Upper ReLU](../../UpperRelu),
and [fc_elu] uses the [exponential linear unit (ELU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELUs).

*** No-FCL CNNs (sim_fc_10\*)
These networks with [no FCL](../../cnn_no_fcl) use various approaches to calculate
accuracy and loss.  Both [sim_fc_10](./sim_fc_10.py) and 
[sim_fc_10_bounded](./sim_fc_10_bounded.py) use the 
[normal Euclidean approach](../../cnn_no_fcl), [sim_fc_10_max](./sim_fc_10_max.py)
uses just an average of the minimum and maximum points (this is further explored
in [EucMaxBenchmarking](./EucMaxBenchmarking)), and the two cosine networks use
some cosine similarity.  The difference between the cosine networks is that
[sim_fc_10_cosine](./sim_fc_10_cosine.py) uses compares the output maps from the
network to predefined target maps, while
[sim_fc_10_cosine_learned](./sim_fc_10_cosine_learned.py) uses learned target maps.
