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
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')
#if summary directory exist, delete the previous summaries
#if tf.gfile.Exists(FLAGS.summary_dir):
#	 tf.gfile.DeleteRecursively(FLAGS.summary_dir)
#	 tf.gfile.MakeDirs(FLAGS.summary_dir)


#Parameters
BatchLength=25	#32 images are in a minibatch
Size=[105, 105, 1] #Input img will be resized to this size
NumIteration=200000;
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 5 #number of output classes
NumSupportsPerClass = 2
EvalFreq=100000 #evaluate on every 100th iteration


# Placeholders
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2] ]) #network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)

# Load in data
data_location = '../Omniglot_data/'
InitTrainData = np.load('{}omniglot_train.npy'.format(data_location))
InitTestData = np.load('{}omniglot_train.npy'.format(data_location))

# Functions for organizing data

def get_train_data(train_data, train_size=5, test_size=15, num_classes=5, Size=[105, 105]):

    class_nums = random.sample(range(0, train_data.shape[0]), num_classes)
    train_data = train_data[class_nums]
    train_data = np.reshape(train_data, [train_size*num_classes, Size[0], Size[1]])

    train_labels = np.asarray([idx / train_size for idx in range(train_size * num_classes)])

    return train_data, train_labels

def get_test_data(test_data, train_size=5, test_size=15, num_classes=5, Size=[105,105]):

    class_nums = random.sample(range(0, test_data.shape[0]), num_classes)
    test_data = test_data[class_nums]
    test_data = np.reshape(test_data, [test_size*num_classes, Size[0], Size[1]])


    test_labels = np.asarray([idx / train_size for idx in range(test_size * num_classes)])

    return test_data, test_labels




#the convolutional network
NumKernels = [32,32,32,32]
def MakeConvNet(Input,Size, First=False):
	CurrentInput = Input
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
	print(Supports.shape)





# Define loss and optimizer
with tf.name_scope('loss'):

	#first calculate cosine similarity 
	#between EncodedQuery and everything in Supports
	#(A*B)/(|A||B|)

	# A * B
	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeated, Supports), [2,3,4])
	# |A|
	MagQuery = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeated), [2,3,4]))
	# |B|
	MagSupport = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2,3,4]))
	# result
	CosSim = DotProduct / (MagQuery * MagSupport)


	#reshape to condense supports from the same class to one thing
	CosSim = tf.reshape(CosSim, [NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_sum(CosSim, 1))

	#apply softmax to cosine similarities
	Probabilities = tf.nn.softmax(CosSim)

	
	#TODO  THIS MIGHT NOT WORK
	#TODO
	Loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(OneHotLabels,CosSim))
	#TODO
	#TODO








with tf.name_scope('optimizer'):	
	Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)


with tf.name_scope('accuracy'):	  
	CorrectPredictions = tf.equal(tf.argmax(Probabilities, 1), tf.argmax(OneHotLabels, 1))
	Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))
  


# Initializing the variables
Init = tf.global_variables_initializer()


#create sumamries, these will be shown on tensorboard

#histogram sumamries about the distribution of the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)


#create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
#tf.summary.scalar("accuracy", Accuracy)



SummaryOp = tf.summary.merge_all()


# Launch the session with default graph
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2 #fraction of GPU used

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	print('Initialized')
	
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	Saver = tf.train.Saver()
	
	# Keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1,NumIteration):


		TrainData, TrainLabels = get_train_data(InitTrainData)


		# need to randomly select the query
		# need to randomly select the support elements
		SupportDataList = []
		QueryDataList = []
		QueryLabelList = []
		
		for i in range(BatchLength):


			QueryClass = np.random.randint(NumClasses)
			QueryIndices = np.argwhere(TrainLabels == QueryClass)
			QueryIndex = QueryIndices[0]

			QueryDataList.append(TrainData[QueryIndex])
			QueryLabelList.append(TrainLabels[QueryIndex])

			SupportDataList.append([])

			for j in range(NumClasses):

				if (j == QueryClass):
					SupportDataList[i].append(np.squeeze(TrainData[QueryIndices[1 : 1+NumSupportsPerClass]], axis=1))
				else:
					SupportIndices = np.argwhere(TrainLabels == j)
					SupportDataList[i].append(np.squeeze(TrainData[SupportIndices[0 : NumSupportsPerClass]], axis=1))

		
		QueryData = np.reshape(QueryDataList, [BatchLength,Size[0], Size[1], Size[2]])
		SupportDataList = np.reshape(SupportDataList, [BatchLength, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]])
		Label = np.reshape(QueryLabelList, [BatchLength])

		






		print(1)
		Summary,_,Acc,L, prob = Sess.run([SummaryOp,Optimizer, Accuracy, Loss, Probabilities],
							feed_dict={InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

		print(2)
		#Summary,_,L = Sess.run([SummaryOp,Optimizer, Loss], feed_dict={InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

		
		#print(prob[0])
		#print('Label', Label[0])


		#print loss and accuracy at every 10th iteration
		if (Step%1)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Accuracy:" + str(Acc))
			print("Loss:" + str(L))

		#independent test accuracy
		if not Step % EvalFreq:
			print("\nTesting Independent set:")
			TestData, TestLabels = get_test_data(InitTestData)
			ct = 0
			Acc = 0
			for k in range(0, TestData.shape[0], BatchLength):
				Data = TestData[k:k+BatchLength]
				Labels = TestLabels[k:k+BatchLength]
				acc = Sess.run(Accuracy, 
						feed_dict = {InputData: Data, InputLabels: Labels, SupportData: SupportDataList})
				#print("Independent Batch:", acc)
				#print(type(acc), type(Acc))
				Acc += acc
				ct += 1
			Acc = Acc / ct
			print("Independent Test set:", Acc, '\n')

		
	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


