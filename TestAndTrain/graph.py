#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate

#read in image
data = np.genfromtxt('testdata.csv', delimiter=',', names=['class', 'x', 'y'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.scatter(data['x'], data['y'], c=data['class'])

lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]

ax1.plot(lims, lims, 'k-')

plt.show() # can also do plt.save or something to save the file and then view it later
