#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime
from scipy import misc
import os
import cv2
import math
import itertools

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
#print dt
flags.DEFINE_string('summary_dir', '/tmp/EEG/{}'.format(dt), 'Summaries directory')

# parameters
BatchLength = 32  # 32 images are in a minibatch
Size = 2500
NumIteration = 500
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
NumSupportsPerClass = 2
NumClassesInSubSet = 2
EvalFreq = 50 # evaluate on every 1000th iteration


# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

# Load in EEG data
directory = './PersonData/'
data_in = np.load('{}split_person_data.npy'.format(directory))
labels_in = np.load('{}split_person_labels.npy'.format(directory))

# restructure data to have a train and a test set
NumElementsPerClass = data_in.shape[0] / NumClasses
TrainSize = 15
TestSize = NumElementsPerClass - TrainSize
Data = np.zeros([NumClasses, NumElementsPerClass, Size])
Labels = np.zeros([NumClasses, NumElementsPerClass])
for k in range(NumClasses):
	k_inds = np.argwhere(labels_in==k)[0]
	Data[k] = data_in[k_inds]
	Labels[k] = labels_in[k_inds]
TrainData = Data[:,0:TrainSize]
TrainLabels = Labels[:,0:TrainSize]
TestData = Data[:,TrainSize:TrainSize+TestSize]
TestLabels = Data[:,TrainSize:TrainSize+TestSize]

#randomize order
permutation = np.random.permutation(TrainData.shape[0])
TrainData = TrainData[permutation]
TrainLabels = TrainLabels[permutation]
permutation = np.random.permutation(TestData.shape[0])
TestData = TestData[permutation]
TestLabels = TestLabels[permutation]


def make_support_set(Data, Labels):
	SupportDataList = []
	QueryDataList = []
	QueryLabelList = []

	for i in range(BatchLength):

		QueryClass = np.random.randint(NumClasses)
		QueryIndices = np.argwhere(Labels == QueryClass)
		QueryIndex = QueryIndices[0]

		QueryDataList.append(Data[QueryIndex])
		QueryLabelList.append(Labels[QueryIndex])

		SupportDataList.append([])

		for j in range(NumClasses):
			if (j == QueryClass):
				SupportDataList[i].append(np.squeeze(Data[QueryIndices[1 : 1 + NumSupportsPerClass]], 1))
			else:
				SupportIndices = np.argwhere(Labels == j)
				SupportDataList[i].append(np.squeeze(Data[SupportIndices[0 : NumSupportsPerClass]], 1))


	QueryData = np.reshape(QueryDataList, [BatchLength, Size])
	SupportDataList = np.reshape(SupportDataList, [BatchLength, NumClasses, NumSupportsPerClass, Size])
	SupportDataList = np.transpose(SupportDataList, (0, 2, 1, 3, 4, 5))
	Label = np.reshape(QueryLabelList, [BatchLength])
	return QueryData, SupportDataList, Label

NumKernels = [32, 32, 32]
def MakeConvNet(Input, Size, First = False):
	CurrentInput = Input
	CurrentInput = (CurrentInput / 255.0) - 0.5
	CurrentFilters = 1 # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(len(NumKernels)): # number of layers
		with tf.variable_scope('conv' + str(i)) as varscope:
			#if not First:
				#varscope.reuse_variables()
			NumKernel = NumKernels[i]
			W = tf.get_variable('W',[15,CurrentFilters,NumKernel])

			Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv1d(CurrentInput, W, stride = 15, padding = 'VALID') #VALID, SAME
			ConvResult= tf.add(ConvResult, Bias)

			# ReLU = tf.nn.relu(ConvResult)

			# leaky ReLU
			alpha = 0.01
			ReLU = tf.maximum(alpha * ConvResult, ConvResult)

			CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 5, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID') # this should be 1, 1, 1, 1 for both if the network is CNN friendly

	return CurrentInput

with tf.name_scope('network'):

	encodedQuery = MakeConvNet(InputData, Size, First = True)
	SupportList = []
	QueryList = []

	for i in range(NumClasses):
		for k in range(NumSupportsPerClass):
			SupportList.append(MakeConvNet(SupportData[:, k, i, :, :, :], Size))
			QueryList.append(encodedQuery)

	QueryRepeat = tf.stack(QueryList)
	Supports = tf.stack(SupportList)

# define loss and optimizer
with tf.name_scope('loss'):
	'''calculate cosine similarity between encodedQuery and everything in Supports
	(A*B)/(|A||B|)'''

	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeat, Supports), [2, 3, 4])
	#QueryMag = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeat), [2, 3, 4]))
	SupportsMag = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2, 3, 4]))
	CosSim = DotProduct / tf.clip_by_value(SupportsMag, 1e-10, float("inf"))

	CosSim = tf.reshape(CosSim, [NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_sum(CosSim, 1))

	probs = tf.nn.softmax(CosSim)

	Loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, CosSim))

with tf.name_scope('optimizer'):
		# use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
		#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
		Pred = tf.argmax(probs, 1)
		Correct = tf.argmax(OneHotLabels, 1)
		CorrectPredictions = tf.equal(Pred, Correct)
		Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))

# initializing the variables
Init = tf.global_variables_initializer()

# create sumamries, these will be shown on tensorboard

# histogram sumamries about the distributio nof the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

# create image summary from the first 10 images
#tf.summary.image('images', TrainData[1 : 10, :, :, :], max_outputs = 50)

# create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)

SummaryOp = tf.summary.merge_all()

# limits the amount of GPU you can use so you don't tie up the server
conf = tf.ConfigProto(allow_soft_placement = True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.25

# launch the session with default graph
with tf.Session(config = conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())
	Saver = tf.train.Saver()

	# keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1, NumIteration + 1):

			TrainData, TrainLabels = get_train_data(datalist)
			# need to change make_support_set to create all combinations
			# need to calculate the accuracy for each combination and keep track of which pair of supports produces the max accuracy
			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute teh session
			Summary, _, Acc, L, p, c, cp = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, Pred, Correct, CorrectPredictions],
				feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

			if (Step % 10 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				count = 0
				for i in range(BatchLength):
					TestData, TestLabels = get_test_data(datalist)
					TestData, SuppData, TestLabels = make_support_set(TestData, TestLabels)

					Acc = Sess.run(Accuracy, feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData})
					TotalAcc += Acc
					count += 1
				TotalAcc = TotalAcc / count
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)

	
	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)