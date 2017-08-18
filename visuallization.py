import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
import time

from numpy import arange

if __name__=="__main__":
    argv = sys.argv
    print argv[1]
    file=""
    if len(argv)==2:
        file=sorted(glob.glob(argv[1]+"*"))
        print file
        file= file[len(file)-1]
        print file
    else:
        file = argv[2]
    print file
    data= pd.read_csv(file)
    plt.close("all")
    _, ax1 = plt.subplots(2, sharex=True)
    ax2 = ax1[0].twinx()
    # print data.columns.values
    # print data.ix[3,1:]
    ax1[0].plot(range(len(data.ix[3,1:])), data.ix[3,1:], 'g')

    ax1[0].plot(range(len(data.ix[4,1:])), data.ix[4,1:], 'y')

    ax2.plot(range(len(data.ix[5,1:])), data.ix[5,1:], 'r')

    ax1[0].set_xlabel('iteration')
    ax1[0].set_ylabel('loss')
    ax2.set_ylabel('accuracy')

    ax1[1].plot(range(len(data.ix[0,1:])), data.ix[0,1:], 'g')
    ax1[1].plot(range(len(data.ix[1,1:])), data.ix[1,1:], 'y')
    ax1[1].plot(range(len(data.ix[2,1:])), data.ix[2,1:], 'r')

    ax1[1].set_xlabel('iteration')
    ax1[1].set_ylabel('precentage')
    plt.ion()
    plt.show()
    time.sleep(10)
    plt.close('all')


