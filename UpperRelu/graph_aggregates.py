import numpy as np
import matplotlib.pyplot as plt

#load in data
datalistA = []
datalistB = []
data_loc = './results/'
for k in range(10):
	dat_a = np.loadtxt('{}bnorm_test_acc{}.dat'.format(data_loc,k))
	dat_b = np.loadtxt('{}test_acc_upper{}.dat'.format(data_loc,k))
	
	#cut off data to a standard length
	dat_a = dat_a[0:30,:]
	dat_b = dat_b[0:30,:]
	#dat_a = dat_a[0:3000,:]
	#dat_b = dat_b[0:3000,:]
	
	datalistA.append(dat_a)
	datalistB.append(dat_b)

#reformat and aggregate
datalistA = np.asarray(datalistA)
datalistB = np.asarray(datalistB)
dataA = np.mean(datalistA, 0)
dataB = np.mean(datalistB, 0)


#cut off data to a standard length
dataA = dataA[0:3000,:]
dataB = dataB[0:3000,:]

#plot
plt.plot(dataA[:,0], dataA[:,1],'b-',
		dataB[:,0], dataB[:,1], 'g-', markersize=1.0)

#display
plt.show()
