## The MindWave EEG
<img src="./mindwave.jpg" alt="Image of MindWave" style="width: 50px;"/>
![]("./mindwave.jpg" | width=50)
The MindWave EEG is a lightweight and inexpensive brainwave reader produced by the
company <a href="http://neurosky.com">NeuroSky</a>.  We were able to read in data 
in csv format from the MindWave using an Android application and used python scripts
to parse this data, retrieving just the raw values and splitting it into several
examples.

## Our Three Datasets
The first dataset we worked with was of four different people staring at the same
image for 60 seconds at a time (three times each).  Overall, we obtained a total of
120 samples - 30 from each person.  Due to an problem in our code, we initially
thought this dataset was too easy for our network to learn because it quickly
achieved 100% classification accuracy.  Though we later found this was incorrect (see
below), we decided to create two more difficult datasets to examine.  For these two
datasets, instead of taking data from different people, we recorded one person 
relaxing vs. doing math (*MathRelaxData*) as well as staring at pictures 
of violent storms and idyllic beach scenes (*BeachStormData*).

## Our Network
To attempt to classify the MindWave data, I modified our One-Shot network (see 
[OneShotLearning](../OneShotLearning/)) to use one-dimensional convolutions to match
the one-dimensional input data from the EEG instead of the
[two dimensional convolutions](../README.md) we had been using on images in all of 
our other networks.  The one-shot framework definitely makes the most sense here
because of the limited amount of data we recorded.

## Results
After initially finding that the networks also achieved 100% accuracy ont the two
more difficult datasets, we concluded that there was something wrong with our code.
After significant debugging, we found a small bug that caused the networks to only
use a few of the data examples in the training and testing, allowing the network
to easily learn to distinguish between them.  When we fixed this bug, our results
obviously took a major hit.  Our network was no longer able to achieve much better
than chance (50%) accuracy on the **MathRelax** and **BeachStorm** datasets.
However, the results on the **Person** dataset were somewhat promising. With a two 
person subset of this dataset, the network can guess correctly about 75% of the time 
(vs. 50% chance), and with the full four classes, it can achieve almost 50% accuracy 
(vs. 25% chance).

## Future Work
Though even the performance of our network on the **Person** dataset was not great,
it is somewhat impressive that it can distinguish between people by just processing
the extremely noisy one-dimensional data we took from the inexpensive MindWave.  We
believe that with a higher-precision EEG and a more tightly controlled environment,
better results are possible.  It is also possible that a different network structure
could be used to better classify the data.  We also considered preprocessing our
data with a fast-fourier transform (FFT) to see if that could help with 
classifiation.  There are many potential applications for the use of deep learning on
brainwave data (e.g. personal authentication and helping the paralyzed), so this was
a very interesting topic to study.  I am interested to see the inevitable progress
the deep learning community will make in this field in the coming years and its
applications.
