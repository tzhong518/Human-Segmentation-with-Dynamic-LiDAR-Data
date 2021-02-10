
from keras import backend as K

import h5py
import numpy as np
import glob
import random
import mtutil
from mtutil import pixelwise_categorical_accuracy
from keras import metrics
from keras.models import load_model
from keras.optimizers import Adam
import xml.etree.ElementTree as ET

class DataGenerator_multi_ver02(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.depth = []
        self.labels = []
        self.vel = []
        self.seq_depth = []
        self.seq_labels = []
        self.seq_human_vel = []
        self.seq_vel = []
        self.array_depth = []
        self.array_labels = []
        self.array_vel = []
        self.inputs = []

    def flow_from_directory(self, directory, nb_labels = 2, nb_seq=1, nb_frame = 32, frame_rate = 2):
        h5_directory = directory + '_h5file/'
        all_files = sorted(glob.glob( h5_directory+'/*.h5' ))
        while True:
            seq_length = int(len(all_files)/nb_frame)

            for count in range(nb_seq):
                seq_number = random.randint(0,seq_length-1)
                frame_files = all_files[seq_number*nb_frame:seq_number*nb_frame+nb_frame]
                frame_count = random.randint(0, nb_frame-frame_rate)
                files = frame_files[frame_count:frame_count+frame_rate]
                count_label = 1
                for file in files:
                    h5file = h5py.File(file,'r')
                    lx = np.single(h5file['/depth'].value)
                    lx = lx / 1000 # 1[m] -> 1
                    lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
                    self.depth.append( lx )
                    if count_label == len(files):
                        defect_mask = (lx[:, :, 0] > 0)
                        defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
                        label = np.single(h5file['/label'].value)
                        ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
                        lv = np.single(h5file['/velocity'].value) / 1000 # 1[m] -> 1
                        if np.sum(label>0) > 0:
                            gain = np.sum(label==0)/np.sum(label>0)
                        else:
                            gain = 1
                        for h in range(32):
                            for w in range(1024):
                                if label[h][w] > 0:
                                    ly[h][w][1] = gain
                                else:
                                    ly[h][w][0] = 1	
                        mask = (ly[:,:,1] > 0)
                        mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
                    else:
                        count_label += 1
                lv = lv * defect_mask
                self.depth.append( mask )
                self.depth.append( defect_mask )
                self.array_depth = np.asarray(self.depth)
                self.seq_labels.append( ly )
                self.seq_vel.append( lv )
                self.seq_human_vel.append( lv*mask )

            inputs = np.asarray(self.array_depth, dtype=np.float32)
            inputs = inputs.reshape( (1, frame_rate+2, 32, 1024, 1) )
            targets00 = np.asarray(self.seq_labels, dtype=np.float32)
            targets01 = np.asarray(self.seq_vel, dtype=np.float32)
            targets02 = np.asarray(self.seq_human_vel, dtype=np.float32)
            targets = [targets00, targets01, targets01, targets02]
            self.reset()
            yield inputs, targets
            
    def flow_from_directory_ver2(self, directory, nb_labels = 2, nb_seq=1, nb_frame = 32, frame_rate = 2):
        h5_directory = directory + '_h5file/'
        all_files = sorted(glob.glob( h5_directory+'/*.h5' ))
        while True:
            seq_length = int(len(all_files)/nb_frame)

            for count in range(nb_seq):
                seq_number = random.randint(0,seq_length-1)
                frame_files = all_files[seq_number*nb_frame:seq_number*nb_frame+nb_frame]
                frame_count = random.randint(0, nb_frame-frame_rate)
                files = frame_files[frame_count:frame_count+frame_rate]
                count_label = 1
                for file in files:
                    h5file = h5py.File(file,'r')
                    lx = np.single(h5file['/depth'].value)
                    lx = lx / 1000 # 1[m] -> 1
                    lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
                    self.depth.append( lx )
                    if count_label == len(files):
                        defect_mask = (lx[:, :, 0] > 0)
                        defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
                        label = np.single(h5file['/label'].value)
                        ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
                        lv = np.single(h5file['/velocity'].value) / 1000 # 1[m] -> 1
                        if np.sum(label>0) > 0:
                            gain = np.sum(label==0)/np.sum(label>0)
                        else:
                            gain = 1
                        for h in range(32):
                            for w in range(1024):
                                if label[h][w] > 0:
                                    ly[h][w][1] = gain
                                else:
                                    ly[h][w][0] = 1	
                        mask = (ly[:,:,1] > 0)
                        mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
                    else:
                        count_label += 1
                lv = lv * defect_mask
                self.depth.append( mask )
                self.depth.append( defect_mask )
                self.array_depth = np.asarray(self.depth)
                self.seq_labels.append( ly )
                self.seq_vel.append( lv )
                self.seq_human_vel.append( lv*mask )

            inputs = np.asarray(self.array_depth, dtype=np.float32)            
            inputs = inputs.reshape( ( nb_seq, frame_rate+2, 32, 1024, 1))
            targets00 = np.asarray(self.seq_labels, dtype=np.float32)
            targets00 = targets00.reshape(( nb_seq,  32, 1024, 2))
            targets01 = np.asarray(self.seq_vel, dtype=np.float32)
            targets01 = targets01.reshape((nb_seq, 32, 1024, 2))
            targets02 = np.asarray(self.seq_human_vel, dtype=np.float32)
            targets02 = targets02.reshape(( nb_seq, 32, 1024,2))
            targets = [targets00, targets01, targets01, targets02]
            self.reset()
            yield inputs, targets
            
    def flow_from_directory_novelo(self, directory, nb_labels = 2, nb_seq=1, nb_frame = 32, frame_rate = 2):
        h5_directory = directory + '_h5file/'
        all_files = sorted(glob.glob( h5_directory+'/*.h5' ))
        while True:
            seq_length = int(len(all_files)/nb_frame)

            for count in range(nb_seq):
                seq_number = random.randint(0,seq_length-1)
                frame_files = all_files[seq_number*nb_frame:seq_number*nb_frame+nb_frame]
                frame_count = random.randint(0, nb_frame-frame_rate)
                files = frame_files[frame_count:frame_count+frame_rate]
                count_label = 1
                for file in files:
                    h5file = h5py.File(file,'r')
                    lx = np.single(h5file['/depth'].value)
                    lx = lx / 1000 # 1[m] -> 1
                    lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
                    self.depth.append( lx )
                    if count_label == len(files):
                        defect_mask = (lx[:, :, 0] > 0)
                        defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
                        label = np.single(h5file['/label'].value)
                        ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
                        if np.sum(label>0) > 0:
                            gain = np.sum(label==0)/np.sum(label>0)
                        else:
                            gain = 1
                        for h in range(32):
                            for w in range(1024):
                                if label[h][w] > 0:
                                    ly[h][w][1] = gain
                                else:
                                    ly[h][w][0] = 1	
                        mask = (ly[:,:,1] > 0)
                        mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
                    else:
                        count_label += 1

                self.depth.append( mask )
                self.depth.append( defect_mask )
                self.array_depth = np.asarray(self.depth)
                self.seq_labels.append( ly )


            inputs = np.asarray(self.array_depth, dtype=np.float32)
            inputs = inputs.reshape( (1, frame_rate+2, 32, 1024, 1) )
            targets00 = np.asarray(self.seq_labels, dtype=np.float32)
            self.reset()
            yield inputs, targets00