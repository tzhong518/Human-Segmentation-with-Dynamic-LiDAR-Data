# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:58:58 2019

@author: tzhong
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from util import load_test, load_rand, load_for_ver02,load_real_for_ver02
from keras import metrics
from keras.models import load_model
from mtutil import pixelwise_categorical_accuracy, croppingwidth
import architecture
import network_2branch
import scipy.misc
from PIL import Image
from matplotlib.pyplot import subplot
import argparse
import os

parser = argparse.ArgumentParser(description='Test function')
parser.add_argument('--test', metavar='test', type=str, 
                    help='test directory')
parser.add_argument('--test_real',metavar='test_real',type=str,
                    help='test_real directory')
parser.add_argument('--model',metavar='model',type=str,
                    help='model name')
parser.add_argument('--network', metavar='network', type=str, 
                    help='network weight')
parser.add_argument('--gpu', metavar='gpu', type=str, 
                    help='gpu number')
parser.add_argument('--frame', metavar='frame', type=int, 
                    help='frame length')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

modelname=args.model
model = getattr(network_2branch, modelname)()
network_dir = os.path.join('../Result',args.network,args.network+'.h5')
model.load_weights(network_dir)
    
def Test():
    X_test, Y_test, V_test, HV_test = load_for_ver02(dir=args.test, frame_rate = args.frame)
    
    prediction = model.predict(X_test, batch_size = 1, verbose = 1)
    
    prediction_segment = prediction[0]
    prediction_velocity = prediction[1]
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    Store_list = []
    s=Y_test.shape
    each_precision = 0
    sum_precision = 0
    threshold = 0.5
    velo_l2norm_GT = []
    velo_l2norm_pred = []
    velo_allMSE = []
    for c1 in range(0,s[0]):
        each_TP = 0
        each_FP = 0
        each_TN = 0
        each_FN = 0
        for c2 in range(0,s[1]):
            for c3 in range(0,s[2]):
                if X_test[c1,-3:-2,c2,c3,0] > 0:
                    if Y_test[c1,c2,c3,1] == 1:
                        if( prediction_segment[c1,c2,c3,1] > threshold ):
                            TP = TP+1
                            each_TP = each_TP+1
                            prediction_segment[c1,c2,c3,1] = 1
                        else:
                            FN = FN + 1
                            each_FN = each_FN+1
                            prediction_segment[c1,c2,c3,1] = 0
                       
                    else:
                        if( prediction_segment[c1,c2,c3,1] > threshold ):
                            FP = FP + 1
                            each_FP = each_FP+1
                            prediction_segment[c1,c2,c3,1] = 1
                        else:
                            TN = TN + 1
                            each_TN = each_TN+1
                            prediction_segment[c1,c2,c3,1] = 0
                    
                else:
                    pass
        try:
            each_precision = each_TP/(each_TP+each_FP)
            each_recall = each_TP/(each_TP+each_FN)

            Store_list.append(np.array([each_precision]))         
        except:
            pass
        
    print('---------Generated data---------')
    print('True positive : ',TP)
    print('False negative : ',FN)
    print('False positive : ',FP)
    print('True negative : ',TN)
    
    print('Accuracy : ',(TP+TN)/(TP+FN+FP+TN))
    A_H = TP/(TP+FN)
    A_B = TN/(TN+FP)
    print('Average Accuracy : ', (A_H+A_B)/2)
    
    Store_list = np.array(Store_list)
    Precision = np.mean(Store_list, axis = 0)[0]
    Recall = np.mean(Store_list, axis = 0)[1]
    
    print('Precision : ', Precision)
    print('Recall : ',Recall)
    print('Threat score :', TP/(TP+FP+FN))
    
    result_array = np.zeros(5)
    result_array[0] = (A_H+A_B)/2
    result_array[1] = Recall
    result_array[2] = TP/(TP+FP+FN)
    np.savetxt('defectless_'+args.test+'-'+args.network[:-5]+'.txt',result_array)

    resultdir = '../Result/'+ args.network

    np.savetxt(resultdir+'defectless_'+args.test+'-'+args.network[:-5]+'.txt',result_array)
    pd.DataFrame(Store_list).to_csv("store_list_5pair2.csv")
    if os.path.exists(resultdir+'/defectless_'+args.test+'-'+'prediction_segment/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test+'-'+'prediction_segment/')
    if os.path.exists(resultdir+'/defectless_'+args.test+'-'+'groundtruth_segment/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test+'-'+'groundtruth_segment/')
    if os.path.exists(resultdir+'/defectless_'+args.test+'-'+'prediction_velocity/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test+'-'+'prediction_velocity/')
    if os.path.exists(resultdir+'/defectless_'+args.test+'-'+'groundtruth_velocity/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test+'-'+'groundtruth_velocity/')
    if os.path.exists(resultdir+'/defectless_'+args.test+'-'+'groundtruth_localization/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test+'-'+'groundtruth_localization/')
    
    prediction_segment[:,:,:,1] = prediction_segment[:,:,:,1] *255
    Y_test[:,:,:,1] = Y_test[:,:,:,1] * 255
    
    for img1 in range(0,prediction_segment.shape[0]):
        im = Image.fromarray(prediction_segment[img1,:,:,1])
        im = im.convert("RGB")
        im.save(resultdir+'/defectless_'+args.test+'-'+'prediction_segment/'+args.network[:-5]+'-depth_0-5-{0:04d}.png'.format(img1))
        im2 = Image.fromarray(Y_test[img1,:,:,1])
        im2 = im2.convert("RGB")
        im2.save(resultdir+'/defectless_'+args.test+'-'+'groundtruth_segment/'+'defectless_'+args.test+'-{0:04d}.png'.format(img1))
    
        np.save(resultdir+'/defectless_'+args.test+'-'+'prediction_velocity/'+args.network[:-5]+'{0:04d}.npy'.format(img1), prediction_velocity[img1, :, :, :])
        np.save(resultdir+'/defectless_'+args.test+'-'+'groundtruth_velocity/'+args.network[:-5]+'{0:04d}.npy'.format(img1), V_test[img1, :, :, :])    
  
    return result_array[2]
    
def Test_in_dist():
    X_test, Y_test, V_test, HV_test = load_for_ver02(dir=args.test, frame_rate = args.frame)
    
    prediction = model.predict(X_test, batch_size = 1, verbose = 1)
    
    prediction_segment = prediction[0]
    prediction_velocity = prediction[1]
    
    sess = tf.Session()
    MSE_Vel = metrics.mse(V_test, prediction_velocity)
    R_MSE_Vel = sess.run(MSE_Vel)
    
    dist_interval = 4
    threatscore_list = []
    
    for dist in range(3):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        Store_list = []
        s=Y_test.shape
        each_precision = 0
        sum_precision = 0
        for c1 in range(0,s[0]):
            each_TP = 0
            each_FP = 0
            each_TN = 0
            each_FN = 0
            for c2 in range(0,s[1]):
                for c3 in range(0,s[2]):
                    if dist < 2:
                        if X_test[c1,-3:-2,c2,c3,0] > dist * dist_interval:
                            if X_test[c1,-3:-2,c2,c3,0] <= (dist+1) * dist_interval:
                                if Y_test[c1,c2,c3,1] == 1:
                                    if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                        TP = TP+1
                                        each_TP = each_TP+1
                                        prediction_segment[c1,c2,c3,1] = 1
                                    else:
                                        FN = FN + 1
                                        each_FN = each_FN+1
                                        prediction_segment[c1,c2,c3,1] = 0
    
                                else:
                                    if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                        FP = FP + 1
                                        each_FP = each_FP+1
                                        prediction_segment[c1,c2,c3,1] = 1
                                    else:
                                        TN = TN + 1
                                        each_TN = each_TN+1
                                        prediction_segment[c1,c2,c3,1] = 0
                    elif dist == 2:
                        if X_test[c1,-3:-2,c2,c3,0] > dist * dist_interval:
                            if Y_test[c1,c2,c3,1] == 1:
                                if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                    TP = TP+1
                                    each_TP = each_TP+1
                                    prediction_segment[c1,c2,c3,1] = 1
                                else:
                                    FN = FN + 1
                                    each_FN = each_FN+1
                                    prediction_segment[c1,c2,c3,1] = 0
    
                            else:
                                if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                    FP = FP + 1
                                    each_FP = each_FP+1
                                    prediction_segment[c1,c2,c3,1] = 1
                                else:
                                    TN = TN + 1
                                    each_TN = each_TN+1
                                    prediction_segment[c1,c2,c3,1] = 0
    
            try:
                each_precision = each_TP/(each_TP+each_FP)
                each_recall = each_TP/(each_TP+each_FN)
                Store_list.append(np.array([each_precision, each_recall]))
            except:
                pass
    
        result_array = np.zeros(3)
        try:
            A_H = TP/(TP+FN)
            A_B = TN/(TN+FP)
    
            Store_list = np.array(Store_list)
            Precision = np.mean(Store_list, axis = 0)[0]
            Recall = np.mean(Store_list, axis = 0)[1]
    
            result_array = np.zeros(3)
            result_array[0] = (A_H+A_B)/2
            result_array[1] = Recall
            result_array[2] = TP/(TP+FP+FN)
    
            threatscore_list.extend(result_array)
            resultdir = os.path.join('../Result',  args.network, args.test+'-'+args.network[:-5]+'_{0:01d}'.format(dist)+'0m'+'.txt')
            np.savetxt(resultdir, result_array)
        except:
            np.savetxt(args.test+'-'+args.network[:-5]+'_{0:01d}'.format(dist)+'0m'+'.txt',result_array)
    
    
    if os.path.exists('./'+args.test+'-'+'prediction_segment/') == False:
        os.mkdir('./'+args.test+'-'+'prediction_segment/')
    if os.path.exists('./'+args.test+'-'+'groundtruth_segment/') == False:
        os.mkdir('./'+args.test+'-'+'groundtruth_segment/')
    if os.path.exists('./'+args.test+'-'+'prediction_velocity/') == False:
        os.mkdir('./'+args.test+'-'+'prediction_velocity/')
    if os.path.exists('./'+args.test+'-'+'groundtruth_velocity/') == False:
        os.mkdir('./'+args.test+'-'+'groundtruth_velocity/')
    if os.path.exists('./'+args.test+'-'+'groundtruth_localization/') == False:
        os.mkdir('./'+args.test+'-'+'groundtruth_localization/')
    
    prediction_segment[:,:,:,1] = prediction_segment[:,:,:,1] *255
    Y_test[:,:,:,1] = Y_test[:,:,:,1] * 255
    
    for img1 in range(0,prediction_segment.shape[0]):
        im = Image.fromarray(prediction_segment[img1,:,:,1])
        im = im.convert("RGB")
        im.save('./'+args.test+'-'+'prediction_segment/'+args.network[:-5]+'-depth_0-5-{0:04d}.png'.format(img1))
        im2 = Image.fromarray(Y_test[img1,:,:,1])
        im2 = im2.convert("RGB")
        im2.save('./'+args.test+'-'+'groundtruth_segment/'+args.test+'-{0:04d}.png'.format(img1))
    
        np.save('./'+args.test+'-'+'prediction_velocity/'+args.network[:-5]+'{0:04d}.npy'.format(img1), prediction_velocity[img1, :, :, :])
        np.save('./'+args.test+'-'+'groundtruth_velocity/'+args.network[:-5]+'{0:04d}.npy'.format(img1), V_test[img1, :, :, :])
    
    return threatscore_list

def Test_with_real():
    X_test, Y_test = load_real_for_ver02(dir=args.test_real, frame_rate = args.frame)
    
    prediction = model.predict(X_test, batch_size = 1, verbose = 1)
    
    prediction_segment = prediction[0]
    prediction_velocity = prediction[1]
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    Store_list = []
    s=Y_test.shape
    each_precision = 0
    sum_precision = 0
    for c1 in range(0,s[0]):
        each_TP = 0
        each_FP = 0
        each_TN = 0
        each_FN = 0
        for c2 in range(0,s[1]):
            for c3 in range(0,s[2]):
                if X_test[c1,-3:-2,c2,c3,0] > 0:
                    if Y_test[c1,c2,c3,1] == 1:
                        if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                            TP = TP+1
                            each_TP = each_TP+1
                            prediction_segment[c1,c2,c3,1] = 1
                        else:
                            FN = FN + 1
                            each_FN = each_FN+1
                            prediction_segment[c1,c2,c3,1] = 0
    
                    else:
                        if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                            FP = FP + 1
                            each_FP = each_FP+1
                            prediction_segment[c1,c2,c3,1] = 1
                        else:
                            TN = TN + 1
                            each_TN = each_TN+1
                            prediction_segment[c1,c2,c3,1] = 0
                else:
                    pass
        try:
            each_precision = each_TP/(each_TP+each_FP)
            each_recall = each_TP/(each_TP+each_FN)
            Store_list.append(np.array([each_precision, each_recall]))
        except:
            pass
    
    print('----------------Real data----------------')
    print('True positive : ',TP)
    print('False negative : ',FN)
    print('False positive : ',FP)
    print('True negative : ',TN)
    
    print('Accuracy : ',(TP+TN)/(TP+FN+FP+TN))
    A_H = TP/(TP+FN)
    A_B = TN/(TN+FP)
    print('Average Accuracy : ', (A_H+A_B)/2)
    
    Store_list = np.array(Store_list)
    Precision = np.mean(Store_list, axis = 0)[0]
    Recall = np.mean(Store_list, axis = 0)[1]
    
    print('Precision : ', Precision)
    print('Recall : ',Recall)
    print('Threat score :', TP/(TP+FP+FN))
    
    result_array = np.zeros(5)
    result_array[0] = (A_H+A_B)/2
    result_array[1] = Recall
    result_array[2] = TP/(TP+FP+FN)
    np.savetxt(resultdir+'defectless_'+args.test_real+'-'+args.network[:-5]+'.txt',result_array)

    resultdir = '../Result/'+ args.network    
  
    if os.path.exists(resultdir+'/defectless_'+args.test_real+'-'+'prediction_segment/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test_real+'-'+'prediction_segment/')
    if os.path.exists(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_segment/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_segment/')
    if os.path.exists(resultdir+'/defectless_'+args.test_real+'-'+'prediction_velocity/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test_real+'-'+'prediction_velocity/')
    if os.path.exists(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_velocity/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_velocity/')
    if os.path.exists(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_localization/') == False:
        os.mkdir(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_localization/')
    
    prediction_segment[:,:,:,1] = prediction_segment[:,:,:,1] *255
    Y_test[:,:,:,1] = Y_test[:,:,:,1] * 255
    
    for img1 in range(0,prediction_segment.shape[0]):
        im = Image.fromarray(prediction_segment[img1,:,:,1])
        im = im.convert("RGB")
        im.save(resultdir+'/defectless_'+args.test_real+'-'+'prediction_segment/'+args.network[:-5]+'-depth_0-5-{0:04d}.png'.format(img1))
        im2 = Image.fromarray(Y_test[img1,:,:,1])
        im2 = im2.convert("RGB")
        im2.save(resultdir+'/defectless_'+args.test_real+'-'+'groundtruth_segment/'+'defectless_'+args.test+'-{0:04d}.png'.format(img1))
    
    return result_array[2]

def Test_with_real_in_dist():
    X_test, Y_test = load_real_for_ver02(dir=args.test_real, frame_rate = args.frame)

    prediction = model.predict(X_test, batch_size = 1, verbose = 1)
    
    prediction_segment = prediction[0]
    prediction_velocity = prediction[1]
    
    dist_interval = 4
    threatscore_list = []
    for dist in range(3):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        Store_list = []
        s=Y_test.shape
        each_precision = 0
        sum_precision = 0
        for c1 in range(0,s[0]):
            each_TP = 0
            each_FP = 0
            each_TN = 0
            each_FN = 0
            for c2 in range(0,s[1]):
                for c3 in range(0,s[2]):
                    if dist < 2:
                        if X_test[c1,-3:-2,c2,c3,0] > dist * dist_interval:
                            if X_test[c1,-3:-2,c2,c3,0] <= (dist+1) * dist_interval:
                                if Y_test[c1,c2,c3,1] == 1:
                                    if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                        TP = TP+1
                                        each_TP = each_TP+1
                                        prediction_segment[c1,c2,c3,1] = 1
                                    else:
                                        FN = FN + 1
                                        each_FN = each_FN+1
                                        prediction_segment[c1,c2,c3,1] = 0
    
                                else:
                                    if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                        FP = FP + 1
                                        each_FP = each_FP+1
                                        prediction_segment[c1,c2,c3,1] = 1
                                    else:
                                        TN = TN + 1
                                        each_TN = each_TN+1
                                        prediction_segment[c1,c2,c3,1] = 0
                    elif dist == 2:
                        if X_test[c1,-3:-2,c2,c3,0] > dist * dist_interval:
                            if Y_test[c1,c2,c3,1] == 1:
                                if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                    TP = TP+1
                                    each_TP = each_TP+1
                                    prediction_segment[c1,c2,c3,1] = 1
                                else:
                                    FN = FN + 1
                                    each_FN = each_FN+1
                                    prediction_segment[c1,c2,c3,1] = 0
    
                            else:
                                if( prediction_segment[c1,c2,c3,1] > 0.5 ):
                                    FP = FP + 1
                                    each_FP = each_FP+1
                                    prediction_segment[c1,c2,c3,1] = 1
                                else:
                                    TN = TN + 1
                                    each_TN = each_TN+1
                                    prediction_segment[c1,c2,c3,1] = 0
    
            try:
                each_precision = each_TP/(each_TP+each_FP)
                each_recall = each_TP/(each_TP+each_FN)
                Store_list.append(np.array([each_precision, each_recall]))
            except:
                pass
    
        result_array = np.zeros(3)
        try:
            A_H = TP/(TP+FN)
            A_B = TN/(TN+FP)
    
            Store_list = np.array(Store_list)
            Precision = np.mean(Store_list, axis = 0)[0]
            Recall = np.mean(Store_list, axis = 0)[1]
    
            result_array = np.zeros(3)
            result_array[0] = (A_H+A_B)/2
            result_array[1] = Recall
            result_array[2] = TP/(TP+FP+FN)
    
            threatscore_list.extend(result_array)
            resultdir = os.path.join('../Result',  args.network, args.test+'-'+args.network[:-5]+'_{0:01d}'.format(dist)+'0m'+'.txt')
            np.savetxt(resultdir, result_array)
        except:
            np.savetxt(args.test_real+'-'+args.network[:-5]+'_{0:01d}'.format(dist)+'0m'+'.txt',result_array)
    
    
    if os.path.exists('./'+args.test_real+'-'+'prediction_segment/') == False:
        os.mkdir('./'+args.test_real+'-'+'prediction_segment/')
    if os.path.exists('./'+args.test_real+'-'+'groundtruth_segment/') == False:
        os.mkdir('./'+args.test_real+'-'+'groundtruth_segment/')
    if os.path.exists('./'+args.test_real+'-'+'prediction_velocity/') == False:
        os.mkdir('./'+args.test_real+'-'+'prediction_velocity/')
    if os.path.exists('./'+args.test_real+'-'+'groundtruth_velocity/') == False:
        os.mkdir('./'+args.test_real+'-'+'groundtruth_velocity/')
    if os.path.exists('./'+args.test_real+'-'+'groundtruth_localization/') == False:
        os.mkdir('./'+args.test_real+'-'+'groundtruth_localization/')
  
    prediction_segment[:,:,:,1] = prediction_segment[:,:,:,1] *255
    Y_test[:,:,:,1] = Y_test[:,:,:,1] * 255
    
    for img1 in range(0,prediction_segment.shape[0]):
        im = Image.fromarray(prediction_segment[img1,:,:,1])
        im = im.convert("RGB")
        im.save('./'+args.test_real+'-'+'prediction_segment/'+args.network[:-5]+'-depth_0-5-{0:04d}.png'.format(img1))
        im2 = Image.fromarray(Y_test[img1,:,:,1])
        im2 = im2.convert("RGB")
        im2.save('./'+args.test_real+'-'+'groundtruth_segment/'+args.test+'-{0:04d}.png'.format(img1))
    
        np.save('./'+args.test_real+'-'+'prediction_velocity/'+args.network[:-5]+'{0:04d}.npy'.format(img1), prediction_velocity[img1, :, :, :])
        
    return threatscore_list

threatscore_Test = Test()
threatscore_list = []
threatscore_list.extend(Test_in_dist())
threatscore_Real = Test_with_real()
threatscore_list.extend(Test_with_real_in_dist())
threatscore_list.extend([0, threatscore_Test, threatscore_list[2], threatscore_list[5], threatscore_list[8], threatscore_Real, threatscore_list[11], threatscore_list[14], threatscore_list[17]])
print(threatscore_list)

threatscore_list = np.asarray(threatscore_list)
threatscore_list = np.around(threatscore_list, decimals = 4)
threatscore_list = threatscore_list.reshape((-1,1))
resultdir = os.path.join('../Result',  args.network, 'Test_in_dist.csv')
pd.DataFrame(threatscore_list).to_csv(resultdir)