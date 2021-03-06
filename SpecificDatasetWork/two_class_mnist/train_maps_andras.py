from __future__ import print_function
import tensorflow as tf
import datetime
import numpy as np
import random


#set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')
#if summary directory exist, delete the previous summaries
if tf.gfile.Exists(FLAGS.summary_dir):
	tf.gfile.DeleteRecursively(FLAGS.summary_dir)
	tf.gfile.MakeDirs(FLAGS.summary_dir)



#Parameters
BatchLength=25 #32 images are in a minibatch
Size=[28, 28, 1] #Input img will be resized to this size
NumIteration=200000;
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 2 #number of output classes
EvalFreq=100 #evaluate on every 100th iteration

#load data
directory = '../../MNIST_data/'
TrainData= np.load('{}partial_train_images.npy'.format(directory))
TrainLabels=np.load('{}partial_train_labels.npy'.format(directory))
TestData= np.load('{}partial_test_images.npy'.format(directory))
TestLabels=np.load('{}partial_test_labels.npy'.format(directory))


#reformat data to our needs
TrainLabels[TrainLabels == 6] = 0
TrainLabels[TrainLabels == 9] = 1
TestLabels[TestLabels == 6] = 0
TestLabels[TestLabels == 9] = 1

# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [BatchLength, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [BatchLength]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)
KeepProb = tf.placeholder(tf.float32) #dropout (keep probability -currently not used)

NumKernels = [4,4,4,1]
def MakeConvNet(Input,Size):
	CurrentInput = Input
	CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
	for i in range(4): #number of layers
		with tf.variable_scope('conv'+str(i)):
			NumKernel=NumKernels[i]
			# W = tf.get_variable('W',[3,3,CurrentFilters,NumKernel])
			W =tf.Variable(tf.random_normal([3,3,CurrentFilters,NumKernel], stddev=0.1), name="W")
			#Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.0))
	
			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
			#ConvResult= tf.add(ConvResult, Bias)
			#add batch normalization
			#beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))
			#gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))
			#Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
			#PostNormalized = tf.nn.batch_normalization(ConvResult,Mean,Variance,beta,gamma,1e-10)

			ReLU = tf.nn.relu(ConvResult)
			#leaky ReLU
			#alpha=0.01
			#ReLU=tf.maximum(alpha*ConvResult,ConvResult)

			CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')

	return CurrentInput

	
# Construct model
OutMaps = MakeConvNet(InputData, Size)

OutShape= OutMaps.get_shape()
print(OutShape)


# Define loss and optimizer
with tf.name_scope('loss'):
		LabelIndices=tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.argmax(OneHotLabels,1),1),1),1)  #32
		GTMap= tf.tile(LabelIndices,tf.stack([1,OutShape[1],OutShape[2],OutShape[3]]) )
		#GTMap= (tf.tile(LabelIndices,tf.stack([1,OutShape[1],OutShape[2],OutShape[3]]) )*2)-1
		print(LabelIndices.shape, OneHotLabels.shape, GTMap.shape)
		#import lakjfnds
		GTMap = tf.cast(GTMap,tf.float32)
		DiffMap=tf.square(tf.subtract(GTMap,OutMaps))
		Loss=tf.reduce_sum(DiffMap)
		#Loss=tf.reduce_mean(DiffMap)
		

with tf.name_scope('optimizer'):	
		#Use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
			#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):	  
		Zeros = tf.zeros(OutShape, tf.float32)
		#Zeros = tf.ones(OutShape, tf.float32)*-1
		Ones = tf.ones(OutShape, tf.float32)
		SquarredDiffZero = tf.square(tf.subtract(OutMaps, Zeros))
		SquarredDiffOne = tf.square(tf.subtract(OutMaps, Ones))
		AvgDiffZero = tf.reduce_mean(SquarredDiffZero, [1,2,3])
		AvgDiffOne = tf.reduce_mean(SquarredDiffOne, [1,2,3])
		Pred = tf.argmin([AvgDiffZero,AvgDiffOne],0)
		CorrectPredictions = tf.equal(tf.cast(Pred,tf.int32), InputLabels)
		Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions,tf.float32))



# Initializing the variables
Init = tf.global_variables_initializer()


#create sumamries, these will be shown on tensorboard

#histogram sumamries about the distributio nof the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

#create image summary from the first 10 images
tf.summary.image('images', TrainData[1:10,:,:,:],  max_outputs=50)

#create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)



SummaryOp = tf.summary.merge_all()


# Launch the session with default graph
with tf.Session() as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	Saver = tf.train.Saver()
	
	Step = 1
	# Keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:
		
		#create train batch - select random elements for training
		TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
		Data=TrainData[TrainIndices,:,:,:]
		Label=TrainLabels[TrainIndices]
		Label=np.reshape(Label,(BatchLength))
		

		#execute the session
		Summary,_,L,A,P = Sess.run([SummaryOp,Optimizer,  Loss,Accuracy,Pred], feed_dict={InputData: Data, InputLabels: Label})
		#train accuracy
		#print("Iteration: "+str(Step))
		#print("Loss:" + str(L))
		#print("Accuracy:" + str(A))
		print("Pred:" + str(P))

		
		#independent test accuracy
		if (Step%EvalFreq)==0:			
			TotalAcc=0;
			Data=np.zeros([BatchLength]+Size)
			for i in range(0,TestData.shape[0],BatchLength):
				if TestData.shape[0] - i < 25:
					break
				Data=TestData[i:(i+BatchLength)]
				Label=TestLabels[i:(i+BatchLength)]
				P = Sess.run(Pred, feed_dict={InputData: Data})
				for i in range(len(P)):
					if P[i]==Label[i]:
						TotalAcc+=1
			print("Iteration: " + str(Step))
			print("Loss: " + str(L))
			print("Accuracy: " + str(A))
			print("Independent Test set: "+str(float(TotalAcc)/TestData.shape[0]))
		#print("Loss:" + str(L))
		
		SummaryWriter.add_summary(Summary,Step)
		Step+=1

	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/model/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


