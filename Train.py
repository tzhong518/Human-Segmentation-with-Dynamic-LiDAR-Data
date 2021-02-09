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
import mtutil
from mtutil import pixelwise_categorical_accuracy, croppingwidth, load_for_ver02
from generator import DataGenerator_multi_ver02
import architecture, network_frame12, network_3d, network_compare, network_2branch, network_multiunet, network_sequnet
from mtcallbacks import CSVLoggerTimestamp
from datetime import datetime
import argparse
from keras.utils import plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import sys
import time
import json
from lookahead import Lookahead

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--gpu', metavar='gpu', type=str, 
                    help='gpu number')
parser.add_argument('--fig', metavar='fig', type=str, 
                    help='whether draw the network')
parser.add_argument('--model', metavar='model name', type=str, 
                    help='model name')
args = parser.parse_args()
print(args)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

#modelname = 'network_frame04_' + args.model
modelname = args.model
#model = getattr(architecture, modelname)()
#model = getattr(network_frame12, modelname)() 
#model = getattr(network_3d, modelname)() 
#model = getattr(network_compare, modelname)()
model = getattr(network_2branch, modelname)()
#model = getattr(network_multiunet, modelname)()
#model = getattr(network_sequnet, modelname)()
#model = network_frame12.network_frame04_notpair_2d()
if args.fig is not None:
    model.summary()
    tofile=modelname+'.png'
    plot_model(model, to_file=tofile,show_shapes=True,show_layer_names=True)

else:
    lr = 3e-5
    decay = 1e-3
    opt = Adam(lr, decay) #lr *= (1. / (1. + self.decay * self.iterations))
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        #weights = np.ones(2)
        weights = K.variable(weights)
        
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, axis=-1)
            return loss
        return loss
        
    def compute_smooth_loss():
        """
        L1-norm on second-order gradient
        """
        def smooth_loss(y_true, y_pred):
            def gradient(y_pred):
                D_dy = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
                D_dx = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
                return D_dx, D_dy
            dx, dy = gradient(y_pred)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss = tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2))
            return loss
        return smooth_loss
        
    #weights = np.ones((2,))   
    weights = np.asarray([10,1])
    mloss = {
        'segment' : 'categorical_crossentropy',
        #'segment' : weighted_categorical_crossentropy(weights),
        'velocity' : 'mean_squared_error',
        #'velocity' : compute_smooth_loss(),
        'dprocess' : 'mean_squared_error',
        'wvelocity' : 'mean_squared_error'
    }
    mlossWeights = {
        'segment' : 1e5,
        'velocity' : 0.0,
        'dprocess' : 1,
        'wvelocity' : 1e3
    }
    
    model.compile(loss=mloss, optimizer=opt, loss_weights=mlossWeights, metrics=[metrics.categorical_crossentropy, pixelwise_categorical_accuracy, metrics.mae])
    print( model.metrics_names )

    train_dir = '../Data/Train2'
    train_datagen = DataGenerator_multi_ver02()
    val_dir = '../Data/Val2'
    val_datagen = DataGenerator_multi_ver02()
    #X_test, Y_test, V_test, HV_test = load_for_ver02(dir='Test', frame_rate = 4)

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
    #tbCallBack = TensorBoard(log_dir='./logs',  # log 逶ｮ蠖・#                 histogram_freq=1,  # 謖臥・菴慕ｭ蛾｢醍紫・・poch・画擂隶｡邂礼峩譁ｹ蝗ｾ・・荳ｺ荳崎ｮ｡邂・##                  batch_size=32,     # 逕ｨ螟壼､ｧ驥冗噪謨ｰ謐ｮ隶｡邂礼峩譁ｹ蝗ｾ
    #                 write_graph=True,  # 譏ｯ蜷ｦ蟄伜お鄂醍ｻ懃ｻ捺桷蝗ｾ
    #                 write_grads=True, # 譏ｯ蜷ｦ蜿ｯ隗・喧譴ｯ蠎ｦ逶ｴ譁ｹ蝗ｾ
    #                 write_images=True,# 譏ｯ蜷ｦ蜿ｯ隗・喧蜿よ焚
    #                 embeddings_freq=0, 
    #                 embeddings_layer_names=None, 
    #                 embeddings_metadata=None)
    #callbacks = [ModelCheckpoint(title+'.hdf5', monitor='val_categorical_crossentropy', verbose=1, save_best_only=True,  mode='min'), CSVLogger(title+'.csv')]
    checkfilepath = savefold + modelname +'{epoch:03d}.h5' 
    checkpoint = ModelCheckpoint(filepath=checkfilepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
                             #period=100)
    #model.fit_generator(generator = train_datagen.flow_from_directory(directory = train_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=4), steps_per_epoch = step, epochs = epochs, verbose = 2, callbacks=callbacks)#, validation_data = (X_test, Y_test))
    #model.fit_generator(generator = train_datagen.flow_from_directory(directory = train_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=4), steps_per_epoch = step, epochs = epochs, verbose = 2, callbacks = [csv_logger_t])#, validation_data = (X_test, Y_test))	
    #model.fit_generator(generator = train_datagen.flow_from_directory(directory = train_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=4), steps_per_epoch = step, epochs = epochs, verbose = 2, callbacks = [csv_logger_t, checkpoint], validation_data = val_datagen.flow_from_directory(directory = val_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=4),validation_steps=50)	
    model.fit_generator(generator = train_datagen.flow_from_directory_ver2(directory = train_dir, nb_labels = nb_classes, nb_seq = nb_seq, frame_rate=16), steps_per_epoch = step, epochs = epochs, verbose = 2, callbacks = [csv_logger_t, checkpoint], validation_data = val_datagen.flow_from_directory(directory = val_dir, nb_labels = nb_classes, nb_seq = 1, frame_rate=16),validation_steps=50)	
    lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
    lookahead.inject(model) # add into model
    model.save(savefold+modelname+date+'.h5')

