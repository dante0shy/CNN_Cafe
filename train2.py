import caffe
from caffe.net_spec import layers, params
from numpy import ceil, arange
from numpy.ma import zeros
from create_net import build_net
from create_solver import create_solver
from do_solve import do_solve
# from pylab import *
#  import os
import psutil
import GPU
import matplotlib.pyplot as plt
import log
import datetime


class trainCaffeNet :
    niter = 6000
    test_interval = 100
    display = 100

    idx = int(ceil(niter * 1.0 / display))
    train_loss = zeros(idx)
    cpu_u = zeros(idx)
    gpu_u = zeros(idx)
    gpu_m_u = zeros(idx)
    test_loss = zeros(idx)
    test_acc = zeros(idx)

    def __init__(self):
        self.time_s = datetime.datetime.now()
        caffe.set_device(0)
        caffe.set_mode_gpu()

    def train(self):
        solver = caffe.get_solver('./model/solver.prototxt')
        _train_loss =0
        _test_loss = 0
        _accuracy = 0
        _cpu = 0
        _gpu = 0
        _gpu_m = 0

        print 'Running solvers for %d iterations...' % self.niter
        for it in range(self.niter):
            solver.step(1)
            _train_loss += solver.net.blobs['loss'].data
            _cpu += psutil.cpu_percent()
            _gpu += GPU.readl(9)
            _gpu_m += GPU.readl(10)
            if it % 100==0 :
                print(str(it)+" CPU : "+str(psutil.cpu_percent()))+'%'
                print str(it) + " GPU : " + str(GPU.readl(9))+'%'
                print str(it) + " GPU_M : " + str(GPU.readl(10))+'%'
                # print(psutil.virtual_memory())  #
                print str(it)+" LOSS : "+ str(_train_loss/100)
                self.train_loss[it / self.test_interval] = _train_loss/100
                self.cpu_u[it / self.test_interval] = _cpu/10000
                self.gpu_u[it / self.test_interval] = _gpu/10000
                self.gpu_m_u[it / self.test_interval] = _gpu_m/10000
                _train_loss = 0
                _cpu = 0
                _gpu = 0
                _gpu_m = 0


            if it % self.test_interval == 0:
                for test_it in range(100):
                    solver.test_nets[0].forward()
                    _test_loss += solver.test_nets[0].blobs['loss'].data
                    _accuracy += solver.test_nets[0].blobs['acc'].data

                    self.test_loss[it / self.test_interval] = _test_loss / 100
                    self.test_acc[it / self.test_interval] = _accuracy / 100
                print str(it)+" VAL_LOSS : "+str(_test_loss / 100)
                print str(it)+" VAL_ACC : "+str( _accuracy / 100)

                _test_loss = 0
                _accuracy = 0
                log.out_log(self.time_s,
                            self.cpu_u,
                            self.gpu_u,
                            self.gpu_m_u,
                            self.train_loss,
                            self.test_loss,
                            self.test_acc)
            # if it % display == 0:
        print 'Done.'
        self.plot(self.cpu_u,
                  self.gpu_u,
                  self.gpu_m_u,
                  self.train_loss,
                  self.test_loss,
                  self.test_acc)
        print 'Out.'

    def plot(self,cpu_u, gpu_u, gpu_m_u,
             train_loss, test_loss, test_acc):
        _, ax1 = plt.subplots(2, sharex=True)
        ax2 = ax1[0].twinx()

        ax1[0].plot(self.display * arange(len(train_loss)),
                    train_loss, 'g')

        ax1[0].plot(self.test_interval * arange(len(test_loss)),
                    test_loss, 'y')

        ax2.plot(self.test_interval * arange(len(test_acc)),
                 test_acc, 'r')

        ax1[0].set_xlabel('iteration')
        ax1[0].set_ylabel('loss')
        ax2.set_ylabel('accuracy')

        ax1[1].plot(self.display * arange(len(cpu_u)), cpu_u, 'g')
        ax1[1].plot(self.display * arange(len(gpu_u)), gpu_u, 'y')
        ax1[1].plot(self.display * arange(len(gpu_m_u)), gpu_m_u, 'r')

        ax1[1].set_xlabel('iteration')
        ax1[1].set_ylabel('precentage')


        plt.show()

if __name__=='__main__':
    trainer= trainCaffeNet()
    trainer.train()



