#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import random
import cv2

for  repeat in range(0,1):
	# set summary dir for tensorflow with FLAGS
	now = datetime.datetime.now()
	dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
	summary_dir= '/tmp/alexnet/cifar/logs/baseline/{}'.format(dt)

	# Parameters
	BatchLength = 32  # 32 images are in a minibatch
	Size = [227, 227, 3]  # Input img will be resized to this size
	NumIteration = 100000
	LearningRate = 1e-4  # learning rate of the algorithm
	NumClasses = 10  # number of output classes
	Dropout = 0.5  # droupout parameters in the FNN layer - currently not used
	EvalFreq = 100  # evaluate on every 100th iteration


	# load data
	directory = '../CIFAR_data/'
	TrainData = np.load('{}Cifar_train_data.npy'.format(directory))
	TrainLabels = np.load('{}Cifar_train_labels.npy'.format(directory))
	TestData = np.load('{}Cifar_test_data.npy'.format(directory))
	TestLabels = np.load('{}Cifar_test_labels.npy'.format(directory))
	"""

	# load data
	directory = './MNIST_data/'
	#directory = '/Users/johnmcguinness/Dropbox/Notre Dame/Budapest/tensorflow_tutorials/MNIST_practice/'
	TrainData = np.load('{}full_train_images.npy'.format(directory))
	TrainLabels = np.load('{}full_train_labels.npy'.format(directory))
	TestData = np.load('{}full_test_images.npy'.format(directory))
	TestLabels = np.load('{}full_test_labels.npy'.format(directory))
	"""


	# Create tensorflow graph
	InputData = tf.placeholder(
	    tf.float32, [None, Size[0], Size[1], Size[2]])  # network input
	InputLabels = tf.placeholder(tf.int32, [None])  # desired network output
	OneHotLabels = tf.one_hot(InputLabels, NumClasses)
	KeepProb = tf.placeholder(tf.float32)  # dropout (keep probability)

	def AddRelUfc(Input):
	    return tf.nn.relu(Input)

	def AddRelUconv(Input):
	    return tf.nn.relu(Input)

	def MakeAlexNet(Input, Size, KeepProb):
	    CurrentInput = Input  # 227,227,3
	    CurrentInput =CurrentInput /255.0
	    with tf.variable_scope('conv1'):
		# first convolution
		W = tf.get_variable('W', [11, 11, 3, 96])
		Bias = tf.get_variable(
		    'Bias', [96], initializer=tf.constant_initializer(0.1))
		ConvResult1 = tf.nn.conv2d(CurrentInput, W, strides=[
		                           1, 4, 4, 1], padding='SAME')  # VALID, SAME
		ConvResult1 = tf.add(ConvResult1, Bias)
		# first relu
		ReLU1 = AddRelUconv(ConvResult1)
		# response normalization
		radius = 2
		alpha = 2e-05
		beta = 0.75
		bias = 1.0
		Norm1 = tf.nn.local_response_normalization(
		    ReLU1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
		# first pooling
		Pool1 = tf.nn.max_pool(Norm1, ksize=[1, 3, 3, 1], strides=[
		                       1, 2, 2, 1], padding='VALID')
	    with tf.variable_scope('conv2'):
		# second convolution
		W = tf.get_variable('W', [5, 5, 96, 256])
		Bias = tf.get_variable(
		    'Bias', [256], initializer=tf.constant_initializer(0.1))
		ConvResult2 = tf.nn.conv2d(
		    Pool1, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
		ConvResult2 = tf.add(ConvResult2, Bias)
		# second relu
		ReLU2 = AddRelUconv(ConvResult2)
		# response normalization
		radius = 2
		alpha = 2e-05
		beta = 0.75
		bias = 1.0
		Norm2 = tf.nn.local_response_normalization(
		    ReLU2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
		# second pooling
		Pool2 = tf.nn.max_pool(Norm2, ksize=[1, 3, 3, 1], strides=[
		                       1, 2, 2, 1], padding='VALID')
	    with tf.variable_scope('conv3'):
		# third convolution
		W = tf.get_variable('W', [3, 3, 256, 384])
		Bias = tf.get_variable(
		    'Bias', [384], initializer=tf.constant_initializer(0.1))
		ConvResult3 = tf.nn.conv2d(
		    Pool2, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
		ConvResult3 = tf.add(ConvResult3, Bias)
		# third relu
		ReLU3 = AddRelUconv(ConvResult3)
	    with tf.variable_scope('conv4'):
		# fourth convolution
		W = tf.get_variable('W', [3, 3, 384, 384])
		Bias = tf.get_variable(
		    'Bias', [384], initializer=tf.constant_initializer(0.1))
		ConvResult4 = tf.nn.conv2d(
		    ReLU3, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
		ConvResult4 = tf.add(ConvResult4, Bias)
		# fourth relu
		ReLU4 = AddRelUconv(ConvResult4)
	    with tf.variable_scope('conv5'):
		# fifth convolution
		W = tf.get_variable('W', [3, 3, 384, 256])
		Bias = tf.get_variable(
		    'Bias', [256], initializer=tf.constant_initializer(0.1))
		ConvResult5 = tf.nn.conv2d(
		    ReLU4, W, strides=[1, 1, 1, 1], padding='SAME')  # VALID, SAME
		ConvResult5 = tf.add(ConvResult5, Bias)
		# fifth relu
		ReLU5 = AddRelUconv(ConvResult5)
		# fifth pooling
		Pool5 = tf.nn.max_pool(ReLU5, ksize=[1, 3, 3, 1], strides=[
		                       1, 2, 2, 1], padding='VALID')
	    with tf.variable_scope('FC1'):
		# first Fully-connected layer
		CurrentShape = Pool5.get_shape()
		FeatureLength = int(
		    CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
		FC = tf.reshape(Pool5, [-1, FeatureLength])
		W = tf.get_variable('W', [FeatureLength, 4096])
		FC = tf.matmul(FC, W)
		Bias = tf.get_variable(
		    'Bias', [4096], initializer=tf.constant_initializer(0.1))
		FC = tf.add(FC, Bias)
		# relu
		FCReLU1 = AddRelUfc(FC)
	    with tf.variable_scope('FC2'):
		# first Fully-connected layer
		FC = tf.reshape(FCReLU1, [-1, 4096])
		W = tf.get_variable('W', [4096, 4096])
		FC = tf.matmul(FC, W)
		Bias = tf.get_variable(
		    'Bias', [4096], initializer=tf.constant_initializer(0.1))
		FC = tf.add(FC, Bias)
		# relu
		FC = tf.nn.dropout(FC, KeepProb)
		FCReLU2 =  tf.nn.relu(FC)
		#FCReLU2 = AddRelUfc(FC)
	    with tf.variable_scope('FC3'):
		# first Fully-connected layer
		FC = tf.reshape(FCReLU2, [-1, 4096])
		W = tf.get_variable('W', [4096, NumClasses])
		FC = tf.matmul(FC, W)
		Bias = tf.get_variable(
		    'Bias', [NumClasses], initializer=tf.constant_initializer(0.1))
		FC = tf.add(FC, Bias)
		# no relu at the end
		Out = FC

	    return Out


	# Construct model
	PredWeights = MakeAlexNet(InputData, Size, KeepProb)


	# Define loss and optimizer
	with tf.name_scope('loss'):
	    Loss = tf.reduce_mean(
		tf.losses.softmax_cross_entropy(OneHotLabels, PredWeights))

	with tf.name_scope('optimizer'):
	    # Use ADAM optimizer this is currently the best performing training algorithm in most cases
	    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
	    #Optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(Loss)
	    #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

	with tf.name_scope('accuracy'):
	    CorrectPredictions = tf.equal(
		tf.argmax(PredWeights, 1), tf.argmax(OneHotLabels, 1))
	    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))


	# Initializing the variables
	Init = tf.global_variables_initializer()


	# create sumamries, these will be shown on tensorboard

	# histogram sumamries about the distribution of the variables
	for v in tf.trainable_variables():
	    tf.summary.histogram(v.name[:-2], v)

	# create image summary from the first 10 images
	tf.summary.image('images', TrainData[1:10, :, :, :],  max_outputs=50)

	# create scalar summaries for lsos and accuracy
	tf.summary.scalar("loss", Loss)
	tf.summary.scalar("accuracy", Accuracy)


	SummaryOp = tf.summary.merge_all()


	# Launch the session with default graph
	conf = tf.ConfigProto(allow_soft_placement=True)
	conf.gpu_options.per_process_gpu_memory_fraction = 0.2  # fraction of GPU used

	with tf.device('/gpu:0'):
	    with tf.Session(config=conf) as Sess:
		Sess.run(Init)
		SummaryWriter = tf.summary.FileWriter(
		    summary_dir, tf.get_default_graph())
		Saver = tf.train.Saver()

		Step = 1
		# Keep training until reach max iterations - other stopping criterion could be added
		while Step < NumIteration:

		    # create train batch - select random elements for training
		    TrainIndices = random.sample(
		        range(TrainData.shape[0]), BatchLength)
		    Data = TrainData[TrainIndices, :, :, :]
		    InData = np.zeros((BatchLength, Size[0], Size[1], Size[2]))
		    Label = TrainLabels[TrainIndices]
		    Label = np.reshape(Label, (BatchLength))
		    #!!!resize the data, this should not be here...just for testing
		    for i in range(BatchLength):
		        #InData[i,:,:,:]= cv2.cvtColor(cv2.resize(Data[i,:,:,:],(227,227)),cv2.COLOR_GRAY2RGB)
		        InData[i, :, :, :] = cv2.resize( Data[i, :, :, :], (227, 227))

		    # execute teh session
		    Summary, _, Acc, L, P = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, PredWeights], feed_dict={
		                                     InputData: InData, InputLabels: Label, KeepProb: Dropout})

		    # print loss and accuracy at every 10th iteration
		    if (Step % 10) == 0:
		        # train accuracy
		        print("Iteration: " + str(Step))
		        print("Accuracy:" + str(Acc))
		        print("Loss:" + str(L))

		    # independent test accuracy
		    if not (Step % EvalFreq) or Step == NumIteration - 1:
		        TotalAcc = 0
		        Data = np.zeros([1] + Size)
		        for i in range(0, TestData.shape[0]):
		            #Data[0] = cv2.cvtColor(cv2.resize(TestData[i],(227,227)),cv2.COLOR_GRAY2RGB)  
		            Data[0] = cv2.resize(TestData[i],(227,227))
		            Label = TestLabels[i]
		            response = Sess.run([PredWeights], feed_dict={
		                                InputData: Data, KeepProb: 1.0})
		            # print(response)
		            if np.argmax(response) == Label:
		                TotalAcc += 1
		        print("Independent Test set: " +
		              str(float(TotalAcc) / TestData.shape[0]))

		    #print("Loss:" + str(L))
		    SummaryWriter.add_summary(Summary, Step)
		    Step += 1

		#print('Saving model...')
		#print(Saver.save(Sess, "./saved/"))

	print("Optimization Finished!")
	print("Execute tensorboard: tensorboard --logdir=" + summary_dir)
