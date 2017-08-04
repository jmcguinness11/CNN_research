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

<img src="http://www.scholarpedia.org/w/images/6/64/CNN_2D.png" width="700">

## CoNNs and CeNNs

## Benchmarking
(Include link to results) (explain why benchmarking)
