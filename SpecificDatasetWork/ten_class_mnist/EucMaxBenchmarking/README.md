Here I explore the what happens if instead of using the normal approach
to [removing the fully connected layer](../../../cnn_no_fcl) from a convolutional
neural network, I simply take the average of the minimum and maximum point from
each output map.

#### Results
When taking into accound an average network startup time of 7.791 seconds,
the min-max version of the network ran an average of 5.91% faster than the
regular Euclidean version while only experiencing about .5% of accuracy degradation.
The min-max version achieved about 97.5% testing accuracy (on MNIST) while the
Euclidean version achieved about 98.0%.

#### Potential Future Exploration
[Cellular neural networks](https://en.wikipedia.org/wiki/Cellular_neural_network)
can find an the average of a map much more efficiently than the minimum or maximum
point, so exploring the effects on accuracy of using the average of the output maps
instead of the average of the minimum and maximum point would be worthwhile.  If
this produces promising results, implementing such a network on a cellular neural
network might be worthwhile.
