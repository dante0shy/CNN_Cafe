'''
Created on 25 Jul 2017

@author: mozat
'''
import os
import numpy as np
import caffe
import cv2
import copy
import glob
import random
import lmdb
import time
from caffe.proto import caffe_pb2 as pb2


class data_process:
    IMAGE_WIDTH = 227
    IMAGE_HEIGHT = 227
    def __init__(self, image_width = 227, image_height = 227):
        self.IMAGE_HEIGHT = image_height
        self.IMAGE_WIDTH = image_width

    def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

        return img


    def make_datum(self,img, label):

        return pb2.Datum(
            channels=3,
            width= self.IMAGE_WIDTH,
            height= self.IMAGE_HEIGHT,
            label=label,
            data=np.rollaxis(img, 2).tostring())

    def load_data(self , time_s):
        #get labels
        label_dict = [x[8:] for x in glob.glob("./train/*")]
        print label_dict

        train_lmdb = './train_lmdb'
        val_lmdb = './val_lmdb'
        os.system('rm -rf  ' + train_lmdb)

        train_data = [img for img in glob.glob("./train/*/*.jpg")]
        random.shuffle(train_data)

        in_db = lmdb.open(train_lmdb, map_size=int(1e12))
        in_db2 = lmdb.open(val_lmdb, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = self.transform_img(img, img_width=self.IMAGE_WIDTH, img_height=self.IMAGE_HEIGHT)
                for dict_idx, label_n in enumerate(label_dict):
                    if label_n in img_path:
                        label = dict_idx
                #print label

                datum1 = self.make_datum(img, label)
                if (in_idx % 10 != 0):
                    in_txn.put('{:0>5d}'.format(in_idx), datum1.SerializeToString())
                    # label_1.append(['{:0>5d}'.format(in_idx), label])

                if in_idx%100==0:
                    print str(time.time() - time_s)+'s : '+'{:0>5d}'.format(in_idx) + ':' + img_path + ' label : ' + str(label)
        in_db.close()
        # with in_db_l.begin(write=True) as in_l_txn:
        #     for i in range(len(label_1)):
        #         in_l_txn.put(label_1[i][0],label_1[i][1])
        # in_db_l.close()

        with in_db2.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = self.transform_img(img, img_width=self.IMAGE_WIDTH, img_height=self.IMAGE_HEIGHT)
                for dict_idx, label_n in enumerate(label_dict):
                    if label_n in img_path:
                        label = dict_idx
                #print label
                # datum1 = 0
                datum1 = self.make_datum(img, label)
                if (in_idx % 10 == 0):
                    in_txn.put('{:0>5d}'.format(in_idx), datum1.SerializeToString())
                    # label_2.append(['{:0>5d}'.format(in_idx), label])

                if in_idx%100==0:
                    print str(time.time() - time_s)+'s : '+'{:0>5d}'.format(in_idx) + ':' + img_path + ' label : ' + str(label)

        in_db2.close()

        # with in_db2_l.begin(write=True) as in_l_txn:
        #     for i in range(len(label_2)):
        #         in_l_txn.put( label_2[i][0],label_2[i][1])
        # in_db2_l.close()
        print 'Finished processing all images'



if __name__ == '__main__':
    time_s = time.time()
    # your code
    data_process().load_data(time_s)
    print time.time() - time_s
