## The MindWave EEG
<img src="./mindwave.jpg" alt="Image of MindWave" style="width: 150px;"/>
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
See [here](../OneShotLearning/)
