

import matplotlib.pyplot as plt
from numpy import arange
import datetime

print str(datetime.datetime.now())
s=[1,1,1,1,1,1,1,1]
_, ax1 = plt.subplots(2, sharex=True)
ax2 = ax1[0].twinx()

ax1[0].plot(1 * arange(len(s)), s, 'g')
s=[1,1,1,1,1,1,1,2]
ax1[0].plot(1 * arange(len(s)), s, 'y')
s=[1,1,1,1,1,2,1,1]
ax2.plot(1 * arange(len(s)), s, 'r')

ax1[0].set_xlabel('iteration')
ax1[0].set_ylabel('loss')
ax2.set_ylabel('accuracy')
s=[1,1,1,1,2,1,1,1]
ax1[1].plot(1 * arange(len(s)), s, 'g')
s=[1,1,1,2,1,1,1,1]
ax1[1].plot(1 * arange(len(s)), s, 'y')
s=[1,1,2,1,1,1,1,1]
ax1[1].plot(1 * arange(len(s)), s, 'r')



ax1[1].set_xlabel('precentage')
ax1[1].set_ylabel('loss')


plt.show()