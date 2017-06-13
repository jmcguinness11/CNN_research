#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import random
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
Size=[28, 28, 1] #Input img will be resized to this size
NumIteration=200000;
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 10 #number of output classes
NumSupportsPerClass = 2 
Dropout=0.5 #droupout parameters in the FNN layer - currently not used
EvalFreq=1000 #evaluate on every 100th iteration

#load data
directory = '../MNIST_data/'
TrainData= np.load('{}full_train_images.npy'.format(directory))
TrainLabels=np.load('{}full_train_labels.npy'.format(directory))
TestData= np.load('{}full_test_images.npy'.format(directory))
TestLabels=np.load('{}full_test_labels.npy'.format(directory))


# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2] ]) #network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)


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
	SupportList = [] 
	QueryList = []
	for i in range(NumClasses):
		same_class_supports = []
		for k in range(NumSupportsPerClass):
			same_class_supports.append(MakeConvNet(SupportData[:,k,i,:,:,:], Size))
		#aggregate support elements from same class
		same_class_supports = tf.stack(same_class_supports)
		SupportList.append(tf.reduce_sum(same_class_supports, 0))
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
	MagQuery = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeated), [2,3,4]))
	# |B|
	MagSupport = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2,3,4]))
	# result
	CosSim = DotProduct / (MagQuery * MagSupport)

	print(CosSim)

	#TODO  THIS MIGHT NOT WORK
	#TODO
	Loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(OneHotLabels,CosSim))
	#TODO
	#TODO

with tf.name_scope('optimizer'):	
	Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

'''
with tf.name_scope('accuracy'):	  
	CorrectPredictions = tf.equal(tf.argmax(PredWeights, 1), tf.argmax(OneHotLabels, 1))
	Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))
'''	  


# Initializing the variables
Init = tf.global_variables_initializer()


#create sumamries, these will be shown on tensorboard

#histogram sumamries about the distribution of the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

#create image summary from the first 10 images
tf.summary.image('images', TrainData[1:10,:,:,:],  max_outputs=50)

#create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
#tf.summary.scalar("accuracy", Accuracy)



SummaryOp = tf.summary.merge_all()


# Launch the session with default graph
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2 #fraction of GPU used

with tf.Session(config=conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	Saver = tf.train.Saver()
	
	# Keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1,NumIteration):
		
		# create train batch - select random elements for training
		#TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
		permutation = np.random.permutation(TrainData.shape[0])
		TrainLabels = TrainLabels[permutation]
		TrainData = TrainData[permutation]
		Data = TrainData[k : k + BatchLength, :, :, :]
		Label = TrainLabels[k : k + BatchLength]
		Label = np.reshape(Label, (BatchLength))

		# need to randomly select the query
		# need to randomly select the support elements
		SupportLabelsList = [[]]
		SupportDataList = [[]]
		QueryDataList = [[]]
		QueryLabelList = [[]]
		for i in range(BatchLength):
			QueryClass = random.randint(0, NumClasses - 1)
			QueryIndices = np.argwhere(TrainLabels == QueryClass)
			QueryIndex = QueryIndices[0]

			QueryDataList.append(TrainData[QueryIndex])
			QueryLabelList.append(TrainLabels[QueryIndex])

			for j in range(NumClasses):
				if (j == QueryClass):
					SupportLabelsList[i].append(TrainLabels[QueryIndices[1 : 3]])
					SupportDataList[i].append(TrainData[QueryIndices[1 : 3]])
				else:
					SupportIndices = np.argwhere(TrainLabels == j)
					SupportDataList[i].append(TrainData[SupportIndices[0 : NumSupportsPerClass]])
					SupportLabelsList[i].append(TrainLabels[SupportIndices[0 : NumSupportsPerClass]])

		QueryData = tf.stack(QueryDataList)
		print(QueryData.shape)

		Summary,_,Acc,L,P = Sess.run([SummaryOp,Optimizer, Accuracy, Loss,PredWeights], feed_dict={InputData: Query, InputLabels: Label})

		#print loss and accuracy at every 10th iteration
		if (Step%100)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Accuracy:" + str(Acc))
			print("Loss:" + str(L))

		#independent test accuracy
		if not (Step%EvalFreq) or Step == NumIteration-1:			
			TotalAcc=0;
			Data=np.zeros([1]+Size)
			for i in range(0,TestData.shape[0]):
				Data[0]=TestData[i]
				Label=TestLabels[i]
				response = Sess.run([PredWeights], feed_dict={InputData: Data})
				#print(response)
				if np.argmax(response)==Label:
					TotalAcc+=1
			print("Independent Test set: "+str(float(TotalAcc)/TestData.shape[0]))
		
		#print("Loss:" + str(L))
		SummaryWriter.add_summary(Summary,Step)

	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


