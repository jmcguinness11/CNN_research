import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


#read in image
InputImage = 'avegra2.bmp'


def CellEquation(Z, t):
	Out = [0, 0]
	Out[0] = Z[1]
	Out[1] = -Z[0]
	return Out
	#implementing x' = y, y' = -x

zinit = [0,7]

a_t = np.arange(0, 25.0, 0.01) #returns np array from 0 to 25 with steps of 0.01


r = scipy.integrate.odeint(CellEquation, zinit, a_t)

print r.shape

plt.plot(r[:,0])
plt.show()
