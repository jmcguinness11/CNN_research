All of the code in this directory works on a subset of MNIST containing only
sixes and nines.

* [train_maps_andras.py](./train_maps_andras.py):
This program is what Andras sent to me after he edited an earlier version of the
program that I worked on.  It is the first program that we worked on to remove the
fully-connected layer from a convolutional neural network.
* [train_maps_1neg1.py](./train_maps_1neg1.py):
This was the first [no-FCL network](../../cnn_no_fcl) in which I used maps of ones
and negative ones instead of ones and zeros.
* [train_maps_1neg1_bounded.py](./train_maps_1neg1_bounded.py):
Same as above except it uses Upper ReLU.
* [train_maps_cosine.py](./train_maps_cosine.py):
This is a version of the above networks that uses cosine similarity to calculate 
accuracy and loss.
* [train_maps_map.py](./train_maps_max.py):
This network simply uses an average of the maximum and minimum points of each of the output maps to calculate accuracy and loss.
