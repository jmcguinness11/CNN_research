#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate
import csv
import pandas as pd

#-------------------

data = np.genfromtxt('testdataFCL.csv', delimiter=',', names=['class', 'x', 'y'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

x = data['x']
y = data['y']
labels = data['class']
df = pd.DataFrame(dict(x = x, y = y, label = labels))

groups = df.groupby('label')

for name, group in groups:
    ax1.plot(group.x, group.y, marker='o', label=name, linestyle='', markersize=2)

lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
reference, = ax1.plot(lims, lims, 'k-', label='Reference Line')

ax1.legend()

plt.show()
