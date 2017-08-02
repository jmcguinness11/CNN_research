#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import random
import os
from scipy import misc
import cv2

#set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
flags.DEFINE_string('summary_dir', '/tmp/oneshot/{}'.format(dt), 'Summaries directory')
#if summary directory exist, delete the previous summaries
#if tf.gfile.Exists(FLAGS.summary_dir):
#	 tf.gfile.DeleteRecursively(FLAGS.summary_dir)
#	 tf.gfile.MakeDirs(FLAGS.summary_dir)


#Parameters
BatchLength=32	#32 images are in a minibatch
#Size=[105, 105, 1] #Input img will be resized to this size
Size=[28,28,1]
NumIteration=200000;
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 10 #number of output classes
NumClassesInSubset = 10
NumSupportsPerClass = 2
TrainSize = 10
TestSize = 10
EvalFreq=20 #evaluate on every 100th iteration


# Placeholders
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2] ]) #network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)

# Read in MNIST data
directory = '../MNIST_data/'
TrainData= np.load('{}full_train_images.npy'.format(directory))
TrainLabels=np.load('{}full_train_labels.npy'.format(directory))
TestData= np.load('{}full_test_images.npy'.format(directory))
TestLabels=np.load('{}full_test_labels.npy'.format(directory))

#randomize order
permutation = np.random.permutation(TrainData.shape[0])
TrainData = TrainData[permutation]
TrainLabels = TrainLabels[permutation]
permutation = np.random.permutation(TestData.shape[0])
TestData = TestData[permutation]
TestLabels = TestLabels[permutation]

#shorten train data so this makes sense as a one-shot problem
TrainDataList = []
TrainLabelList = []
for classnum in range(NumClasses):
	train_indices = np.argwhere(TrainLabels == classnum)[:,0]
	TrainDataList.append(TrainData[train_indices[0:TrainSize]])
	TrainLabelList.append(TrainLabels[train_indices[0:TrainSize]])

print(np.asarray(TrainDataList).shape)
TrainData = np.reshape(TrainDataList, [TrainSize*NumClasses, Size[0], Size[1], Size[2]])
TrainLabels = np.reshape(TrainLabelList, [TrainSize*NumClasses])



#function for creating the support set
def make_support_set(Data, Labels):
	SupportDataList = []
	QueryDataList = []
	QueryLabelList = []
	
	for i in range(BatchLength):


		QueryClass = np.random.randint(NumClasses)
		QueryIndices = np.argwhere(Labels == QueryClass)
		permutation = np.random.permutation(QueryIndices.shape[0])
		QueryIndices = QueryIndices[permutation]
		QueryIndex = QueryIndices[0]

		QueryDataList.append(Data[QueryIndex])
		QueryLabelList.append(Labels[QueryIndex])

		SupportDataList.append([])

		for j in range(NumClasses):

			if (j == QueryClass):
				SupportDataList[i].append(np.squeeze(Data[QueryIndices[1 : 1+NumSupportsPerClass]], axis=1))
			else:
				SupportIndices = np.argwhere(Labels == j)
				SupportDataList[i].append(np.squeeze(Data[SupportIndices[0 : NumSupportsPerClass]], axis=1))


	QueryData = np.reshape(QueryDataList, [BatchLength,Size[0], Size[1], Size[2]])
	
	#reshape and swap dimensions
	SupportDataList = np.reshape(SupportDataList, [BatchLength, NumClasses, NumSupportsPerClass, Size[0], Size[1], Size[2]])
	SupportDataList = np.transpose(SupportDataList, (0, 2, 1, 3, 4, 5))
	Label = np.reshape(QueryLabelList, [BatchLength])
	return QueryData, SupportDataList, Label



#the convolutional network
NumKernels = [32,32,32]
def MakeConvNet(Input,Size, First=False):
	CurrentInput = Input
	CurrentInput=(CurrentInput/255.0)-0.5
	CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
	for i in range(len(NumKernels)): #number of layers
		with tf.variable_scope('conv'+str(i)) as varscope:
			if not First:
				varscope.reuse_variables()
			NumKernel=NumKernels[i]
			W = tf.get_variable('W',[3,3,CurrentFilters,NumKernel])
			Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.1))
		
			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='VALID') #VALID, SAME
			ConvResult= tf.add(ConvResult, Bias)
			
			#add batch normalization
			#beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))
			#gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))
			#Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
			#PostNormalized = tf.nn.batch_normalization(ConvResult,Mean,Variance,beta,gamma,1e-10)
	
			#ReLU = tf.nn.relu(ConvResult)
			#leaky ReLU
			alpha=0.01
			ReLU=tf.maximum(alpha*ConvResult,ConvResult)	
			#ReLU=tf.minimum((1+alpha*(ReLU-1)),ReLU)

			CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,3,3,1],strides=[1,1,1,1],padding='VALID')
			
	return CurrentInput
	



with tf.name_scope('network'):
	EncodedQuery = MakeConvNet(InputData, Size, First=True)
	print('EQ:', EncodedQuery.shape)
	
	SupportList = [] 
	QueryList = []
	for i in range(NumClasses):
		
		for k in range(NumSupportsPerClass):
			SupportList.append(MakeConvNet(SupportData[:,k,i,:,:,:], Size))
			QueryList.append(EncodedQuery) 
		
	
	QueryRepeated = tf.stack(QueryList)
	Supports = tf.stack(SupportList)





# Define loss and optimizer
with tf.name_scope('loss'):
        
	#first calculate cosine similarity 
	#between EncodedQuery and everything in Supports
	#(A*B)/(|A||B|)
        # A * B
	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeated, Supports), [2,3,4])
	# |A|
	#MagQuery = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeated), [2,3,4]))
	# |B|
	MagSupport = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2,3,4]))
	# result
	CosSim = DotProduct / tf.clip_by_value(  MagSupport ,1e-10,float("inf"))

        #reshape to condense supports from the same class to one thing
	CosSim = tf.reshape(CosSim, [ NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_mean(CosSim, 1))

	#apply softmax to cosine similarities
	Probabilities = tf.nn.softmax(CosSim)

	
	#TODO  THIS MIGHT NOT WORK
	#TODO
	Loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(OneHotLabels,Probabilities))
	#TODO
	#TODO



with tf.name_scope('optimizer'):	
	Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)


with tf.name_scope('accuracy'):	 
	Pred = tf.argmax(Probabilities, 1)
	Correct = tf.argmax(OneHotLabels, 1)
	CorrectPredictions = tf.equal(Pred, Correct)
	Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))
  


# Initializing the variables
Init = tf.global_variables_initializer()


#create sumamries, these will be shown on tensorboard

#histogram sumamries about the distribution of the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)


#create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)



SummaryOp = tf.summary.merge_all()


# Launch the session with default graph
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.25 #fraction of GPU used

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	print('Initialized')
	
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	Saver = tf.train.Saver()



	# Keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1,NumIteration+1):

		QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)


		Summary,_,Acc,L, p, c,s = Sess.run([SummaryOp,Optimizer, Accuracy, Loss, Pred, Correct,CosSim],
							feed_dict={InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})
                #print(s)
                #print(c)
		'''
		print(p[0:20])
		print(c[0:20])
		print()
		'''


		#print loss and accuracy at every 10th iteration
		if (Step%5)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Accuracy:" + str(Acc))
			print("Loss:" + str(L))


		#average independent test accuracy
		if not Step % EvalFreq:
			print("\nTesting Independent set:")
			Acc = 0
			ct = 0
			for k in range(25):

				Data, SuppData, Label = make_support_set(TestData, TestLabels)

				acc, p, c = Sess.run([Accuracy, Pred, Correct], 
						feed_dict = {InputData: Data, InputLabels: Label, SupportData: SuppData})
				Acc += acc
				ct = k+1

			Acc /= ct
			print("Independent Test set:", Acc, '\n')

		
	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


