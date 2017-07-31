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
BatchLength=25 #25 images are in a minibatch
Size=[28, 28, 1] #Input img will be resized to this size
NumIteration=250000;
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 10 #number of output classes
EvalFreq=100 #evaluate on every 100th iteration

#load data
directory = '../../MNIST_data/'
TrainData= np.load('{}full_train_images.npy'.format(directory))
TrainLabels=np.load('{}full_train_labels.npy'.format(directory))
TestData= np.load('{}full_test_images.npy'.format(directory))
TestLabels=np.load('{}full_test_labels.npy'.format(directory))




# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [BatchLength, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [BatchLength]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)
KeepProb = tf.placeholder(tf.float32) #dropout (keep probability -currently not used)

NumKernels = [32,32,32,32,10]
def MakeConvNet(Input,Size):
	CurrentInput = Input
	CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
	for i in range(5): #number of layers
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

			#upper ReLU
			alpha=0.01
			ReLU=tf.maximum(-1 + alpha*(ConvResult+1),ConvResult)
			alpha=-0.01
			ReLU=tf.minimum(1 + alpha*(ReLU-1),ReLU)

			CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')

	return CurrentInput

	
# Construct model
OutMaps = MakeConvNet(InputData, Size)

OutShape= OutMaps.get_shape()
print(OutShape)



# Define loss and optimizer
with tf.name_scope('loss'):

	#learned cosine similarity = (A*W)/(|A||W|)
	#define W
	W_cos = tf.Variable(tf.random_normal(OutMaps.shape, stddev=0.1), name="W_cos")
	#A * B
	DotProduct = tf.reduce_sum(tf.multiply(OutMaps, W_cos),[1,2]) #necessary b/c actually -1s
	#|A|
	MagMap = tf.sqrt(tf.reduce_sum(tf.square(OutMaps), [1,2]))
	#|B|
	MagWeights = tf.sqrt(tf.reduce_sum(tf.square(W_cos), [1,2]))
	#result
	CosSim = DotProduct / tf.clip_by_value((MagMap*MagWeights), 1e-10, float("inf"))


	#actual loss calculation
	Probabilities = tf.nn.softmax(CosSim)
	Loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(OneHotLabels,Probabilities))


		

with tf.name_scope('optimizer'):	
		#Use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):	  

	Pred = tf.argmax(CosSim,1)
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
conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.25 #fraction of GPU used

# Launch the session with default graph
with tf.Session(config=conf) as Sess:
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
		Summary,_,L,A,P, labels, CP, probs = Sess.run([SummaryOp,Optimizer, Loss,Accuracy,Pred, InputLabels, CorrectPredictions, Probabilities], feed_dict={InputData: Data, InputLabels: Label})


		if not Step % 20:
			print("Iteration: " + str(Step))
			print("Loss: " + str(L))
			print("Accuracy: " + str(A))
		
		#independent test accuracy
		if (Step%EvalFreq)==0:			
			TotalAcc=0
			Data=np.zeros([BatchLength]+Size)
			for i in range(0,TestData.shape[0],BatchLength):
				if TestData.shape[0] - i < 25:
					break
				Data=TestData[i:(i+BatchLength)]
				Label=TestLabels[i:(i+BatchLength)]
				P = Sess.run(Pred, feed_dict={InputData: Data, InputLabels:Label})
				for i in range(len(P)):
					if P[i]==Label[i]:
						TotalAcc+=1

			print("Independent Test set: "+str(float(TotalAcc)/TestData.shape[0]))
		#print("Loss:" + str(L))
		
		SummaryWriter.add_summary(Summary,Step)
		Step+=1

	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/model/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


