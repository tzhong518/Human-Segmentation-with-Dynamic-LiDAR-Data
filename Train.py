#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

from keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf
from keras import metrics
from keras.optimizers import Adam
from keras import losses
from util import pixelwise_categorical_accuracy, croppingwidth, load_for_ver02
from generator import DataGenerator_multi_ver02
from callbacks import CSVLoggerTimestamp
import network_2branch

import argparse
from keras.utils import plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import sys
import time
import json

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--gpu', metavar='gpu', type=str, 
                    help='gpu number')
parser.add_argument('--fig', metavar='fig', type=str, 
                    help='whether draw the network')
parser.add_argument('--model', metavar='model name', type=str, 
                    help='model name')
parser.add_argument('--frame', metavar='frame', type=int, 
                    help='frame length')
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

frame = args.frame
modelname = args.model
model = getattr(network_2branch, modelname)()

if args.fig is not None:
    model.summary()
    tofile=modelname+'.png'
    plot_model(model, to_file=tofile,show_shapes=True,show_layer_names=True)

else:
    lr = 3e-5
    decay = 1e-3
    opt = Adam(lr, decay) #lr *= (1. / (1. + self.decay * self.iterations))

    mloss = {
        'segment' : 'categorical_crossentropy',
        'velocity' : 'mean_squared_error',
        'dprocess' : 'mean_squared_error',
        'wvelocity' : 'mean_squared_error'
    }
    mlossWeights = {
        'segment' : 1e5,
        'velocity' : 0.0,
        'dprocess' : 1,
        'wvelocity' : 1e3
    }
    
    model.compile(loss=mloss, optimizer=opt, loss_weights=mlossWeights, 
                  metrics=[metrics.categorical_crossentropy, pixelwise_categorical_accuracy, metrics.mae])

    train_dir = '../Data/Train2'
    train_datagen = DataGenerator_multi_ver02()
    val_dir = '../Data/Val2'
    val_datagen = DataGenerator_multi_ver02()

    nb_seq = 1
    nb_classes = 2
    step = 250
    epochs = 1000

    title, ext = os.path.splitext(sys.argv[0])

    date=time.strftime("%m%d", time.localtime()) 
    if os.path.exists('../Result/'+modelname+date) == False:
        os.mkdir('../Result/'+modelname+date)
    savefold='../Result/'+modelname+date+'/'

    trainpara = [{'learningrate':lr, 'decay':decay,'step':step,'epochs':epochs}]
    json_path = os.path.join("../Result",modelname+date,modelname+date+".json")
    with open(json_path, 'w') as f:
        json.dump(trainpara,f)
        
    csv_logger_t = CSVLoggerTimestamp(savefold+modelname+date+'.csv')
    checkfilepath = savefold + modelname +'{epoch:03d}.h5' 
    checkpoint = ModelCheckpoint(filepath=checkfilepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

    model.fit_generator(generator = train_datagen.flow_from_directory_ver2(directory = train_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=frame), steps_per_epoch = step, epochs = epochs, verbose = 2, callbacks = [csv_logger_t, checkpoint], 
                        validation_data = val_datagen.flow_from_directory(directory = val_dir, nb_labels = nb_classes, nb_seq = 1, frame_rate=frame),validation_steps=50)	

    model.save(savefold+modelname+date+'.h5')

