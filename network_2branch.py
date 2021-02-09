# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:31:01 2020

@author: tzhong
"""
from keras.layers.convolutional import Conv3D, UpSampling3D, Conv2D, UpSampling2D
from keras.layers.core import Reshape, Permute, Flatten, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Activation, Lambda, Concatenate, MaxPooling3D, RNN, MaxPooling2D, Multiply
from keras.models import Model
from keras.layers.merge import Add
import math
import numpy as np

from keras import backend as K
K.set_image_data_format('channels_last')

def Expand_Dim_Layer(tensor,axis):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=axis)
    return Lambda(expand_dim)(tensor)

def Squeeze_Layer(tensor,axis):
    def squeeze_dim(tensor):
        return K.squeeze(tensor,axis=axis)
    return Lambda(squeeze_dim)(tensor)

def Padding_Layer(tensor,axis):
    def padding(tensor):
        if axis==-2:
            return K.spatial_3d_padding(tensor,padding=((1,1),(1,1),(0,0)))
        if axis==1:
            return K.spatial_3d_padding(tensor,padding=((0,0),(1,1),(1,1)))
    return Lambda(padding)(tensor)

def R2plus1D_Layer(tensor,in_channels, out_channels,kernel_size,padding):
    intermed_channels = int(math.floor((2* kernel_size * kernel_size * in_channels * out_channels)/ \
                    (kernel_size* kernel_size * in_channels + 2 * out_channels)))
    if padding==False:
        spatialconv=Conv3D(out_channels,(1,kernel_size,kernel_size),padding='same',activation='relu')
        tensor=spatialconv(tensor)
#        temporalconv=Conv3D(out_channels,(2,1,1),padding='same',activation='relu')
    if padding==True:
        spatialconv=Conv3D(intermed_channels,(1,kernel_size,kernel_size),padding='same',activation='relu')
        temporalconv=Conv3D(out_channels,(4,1,1),padding='valid',activation='relu')
        tensor=spatialconv(tensor)
        tensor=temporalconv(tensor)
    return tensor

def network_frame01_unet( input_shape=(3, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    human_mask = Lambda(lambda x: x[:,1,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,2,:,:,:], name = "defect_mask")(input_img)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input00)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input00)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
    spatial = encode

    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    # up_merged = UpSampling2D( (1, 8) )(merged)	
    # process01 = Concatenate(axis = -1)( [up_merged, process10] )
    # process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    # process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    # process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    # process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    # dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    # wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    # outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, output=process10)
    return model
    
def network_frame04_2branch( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model1 = Model(inputs=inputframe, outputs=out)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)
    
#    conv01 = Conv2D(64, (3, 3), padding='same', activation='relu')
#    conv02 = Conv2D(64, (3, 3), padding='same', activation='relu')
#    conv03 = Conv2D(128, (3, 3), padding='same', activation='relu')
#    conv04 = Conv2D(128, (3, 3), padding='same', activation='relu')
#    conv05 = Conv2D(256, (3, 3), padding='same', activation='relu')
#    conv06 = Conv2D(256, (3, 3), padding='same', activation='relu')
#    conv06_split01 = Conv2D(64, (1,1), padding='same', activation='relu')
#    conv07 = Conv2D(512, (3, 3), padding='same', activation='relu')
#    conv08 = Conv2D(512, (3, 3), padding='same', activation='relu')
#    conv08_split01 = Conv2D(64, (1, 1), padding='same', activation='relu')
#    conv08_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')
#    conv09 = Conv2D(512, (3, 3), padding='same', activation='relu')
#    conv10 = Conv2D(512, (3, 3), padding='same', activation='relu')
#    conv10_split01 = Conv2D(64, (1, 1), padding='same', activation='relu')
#    conv10_split02 = Conv2D(64, (1, 5), padding='same', activation='relu')
    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(8, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    spatial = feature_model1(input03)

    merged00 = feature_model2(input00)
#    merged00b = merged0[1]
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

#    merged10 = Concatenate(axis = -1)( [merged00, merged01] )
#    merged11 = Concatenate(axis = -1)( [merged00, merged02] )
#    merged12 = Concatenate(axis = -1)( [merged00, merged03] )

    process00 = UpSampling2D( (1, 2) )(merged_temporal)	
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)

    process10 = Concatenate(axis=-1)([process10,process00])
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = conv12s(process10)
    process10 = conv12bs(process10)
    
    process10 = Concatenate(axis=-1)([process10,process00])
    
    process10 = UpSampling2D((1,2))(process10)
    process10 = conv13(process10) #18
    process10 = conv13b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame04_2branchb( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(8, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Concatenate(axis=-1)([encode_split03, process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Concatenate(axis=-1)([encode_split02, process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Concatenate(axis=-1)([encode_split01, process10, process00])
    process10 = conv13(process10) #18
    process10 = conv13b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame02_2branchd_revised( input_shape=(4, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    human_mask = Lambda(lambda x: x[:,2,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,3,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input01)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input01)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    process01 = Concatenate(axis = -1)( [process00, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame02_2branchd( input_shape=(4, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    human_mask = Lambda(lambda x: x[:,2,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,3,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input01)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input01)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
    
def network_frame04_2branchd( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame04_2branchd_revised( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same')(process01)

    #smooth
    process01 = Multiply(name='velocity')([process01, defect_mask])
    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame08_2branchd_revised( input_shape=(10, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    input04 = Lambda(lambda x: x[:,4,:,:,:], name = "i_divided_4")(input_img)
    input05 = Lambda(lambda x: x[:,5,:,:,:], name = "i_divided_5")(input_img)
    input06 = Lambda(lambda x: x[:,6,:,:,:], name = "i_divided_6")(input_img)
    input07 = Lambda(lambda x: x[:,7,:,:,:], name = "i_divided_7")(input_img)
    human_mask = Lambda(lambda x: x[:,8,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,9,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    merged04 = feature_model2(input04)
    merged05 = feature_model2(input05)
    merged06 = feature_model2(input06)
    merged07 = feature_model2(input07)
    
    #merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    #process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    #process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
    
def network_frame08_2branchd( input_shape=(10, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    input04 = Lambda(lambda x: x[:,4,:,:,:], name = "i_divided_4")(input_img)
    input05 = Lambda(lambda x: x[:,5,:,:,:], name = "i_divided_5")(input_img)
    input06 = Lambda(lambda x: x[:,6,:,:,:], name = "i_divided_6")(input_img)
    input07 = Lambda(lambda x: x[:,7,:,:,:], name = "i_divided_7")(input_img)
    human_mask = Lambda(lambda x: x[:,8,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,9,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    merged04 = feature_model2(input04)
    merged05 = feature_model2(input05)
    merged06 = feature_model2(input06)
    merged07 = feature_model2(input07)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input07)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input07)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
    
def network_frame16_2branchd( input_shape=(18, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    input04 = Lambda(lambda x: x[:,4,:,:,:], name = "i_divided_4")(input_img)
    input05 = Lambda(lambda x: x[:,5,:,:,:], name = "i_divided_5")(input_img)
    input06 = Lambda(lambda x: x[:,6,:,:,:], name = "i_divided_6")(input_img)
    input07 = Lambda(lambda x: x[:,7,:,:,:], name = "i_divided_7")(input_img)
    input08 = Lambda(lambda x: x[:,8,:,:,:], name = "i_divided_8")(input_img)
    input09 = Lambda(lambda x: x[:,9,:,:,:], name = "i_divided_9")(input_img)
    input10 = Lambda(lambda x: x[:,10,:,:,:], name = "i_divided_10")(input_img)
    input11 = Lambda(lambda x: x[:,11,:,:,:], name = "i_divided_11")(input_img)
    input12 = Lambda(lambda x: x[:,12,:,:,:], name = "i_divided_12")(input_img)
    input13 = Lambda(lambda x: x[:,13,:,:,:], name = "i_divided_13")(input_img)
    input14 = Lambda(lambda x: x[:,14,:,:,:], name = "i_divided_14")(input_img)
    input15 = Lambda(lambda x: x[:,15,:,:,:], name = "i_divided_15")(input_img)
    human_mask = Lambda(lambda x: x[:,16,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,17,:,:,:], name = "defect_mask")(input_img)
    
    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    merged04 = feature_model2(input04)
    merged05 = feature_model2(input05)
    merged06 = feature_model2(input06)
    merged07 = feature_model2(input07)
    merged08 = feature_model2(input08)
    merged09 = feature_model2(input09)
    merged10 = feature_model2(input10)
    merged11 = feature_model2(input11)
    merged12 = feature_model2(input12)
    merged13 = feature_model2(input13)
    merged14 = feature_model2(input14)
    merged15 = feature_model2(input15)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07, merged08, merged09, merged10, merged11,merged12, merged13, merged14, merged15])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03, merged04, merged05, merged06, merged07, merged08, merged09, merged10, merged11,merged12, merged13, merged14, merged15])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input07)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input07)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
    
def network_frame04_2branchf( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    encode_split03t = Conv2D(64, (1,1), padding='same', activation='relu')(encode_split03)
    process00 = Concatenate(axis=-1)([encode_split03t, process00])
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = Concatenate(axis=-1)([encode_split02, process00])
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = Concatenate(axis=-1)([encode_split01, process00])
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    #process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    #process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model    

def network_frame04_2branchf_revised( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    encode_split03t = Conv2D(256, (1,1), padding='same', activation='relu')(encode_split03)
    process00 = Add()([encode_split03t, process00])
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = Add()([encode_split02, process00])
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = Add()([encode_split01, process00])
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    #process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    #process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model    

def network_frame04_2branchf_revised2( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03t = Conv2D(256, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    #encode_split03t = Conv2D(256, (1,1), padding='same', activation='relu')(encode_split03)
    process00 = Concatenate(axis=-1)([encode_split03t, process00])
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = Concatenate(axis=-1)([encode_split02, process00])
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = Concatenate(axis=-1)([encode_split01, process00])
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame04_2branchd_noadd( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

#    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
#    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
#    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
#    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
#    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
#    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model

def network_frame04_2branchd_novelo( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    # up_merged = UpSampling2D( (1, 8) )(merged)	
    # process01 = Concatenate(axis = -1)( [up_merged, process10] )
    # process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    # process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    # process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    # process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    # dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    # wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    # outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, output=process10)
    return model

def network_frame04_2branche_slr21( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):

    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)
    
    f_ax = 1

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    
    merged00=Expand_Dim_Layer(merged00,f_ax)
    merged01=Expand_Dim_Layer(merged01,f_ax)
    merged02=Expand_Dim_Layer(merged02,f_ax)
    merged03=Expand_Dim_Layer(merged03,f_ax)    
    
    merged_temporal = Concatenate(axis = f_ax)([merged00, merged01, merged02, merged03])
    merged_temporal = R2plus1D_Layer(merged_temporal,64,128,3,True)
    merged_temporal = Squeeze_Layer(merged_temporal, f_ax) 

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
    
def network_frame04_2branche_slr21b( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)
    
    f_ax = 1

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    
    merged00=Expand_Dim_Layer(merged00,f_ax)
    merged01=Expand_Dim_Layer(merged01,f_ax)
    merged02=Expand_Dim_Layer(merged02,f_ax)
    merged03=Expand_Dim_Layer(merged03,f_ax)    
    
    merged_temporal = Concatenate(axis = f_ax)([merged00, merged01, merged02, merged03])
    merged_temporal = UpSampling3D((1,1,2))(merged_temporal)
    merged_temporal = R2plus1D_Layer(merged_temporal,64,128,3,True)
    merged_temporal = R2plus1D_Layer(merged_temporal,64,128,3,False)
    process00 = Squeeze_Layer(merged_temporal, f_ax) 

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    # process00 = UpSampling2D( (1, 2) )(merged_temporal)
    # process00 = conv11(process00) #16  
    # process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model    
    
def network_frame04_1branch( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)
    
    f_ax = 1

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    
    # merged00=Expand_Dim_Layer(merged00,f_ax)
    # merged01=Expand_Dim_Layer(merged01,f_ax)
    # merged02=Expand_Dim_Layer(merged02,f_ax)
    # merged03=Expand_Dim_Layer(merged03,f_ax)    
    
    # merged_temporal = Concatenate(axis = f_ax)([merged00, merged01, merged02, merged03])
    # merged_temporal = UpSampling3D((1,1,2))(merged_temporal)
    # merged_temporal = R2plus1D_Layer(merged_temporal,64,128,3,True)
    # process00 = Squeeze_Layer(merged_temporal, f_ax) 

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    # process00 = UpSampling2D( (1, 2) )(merged_temporal)
    # process00 = conv11(process00) #16  
    # process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
#    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    # process00 = UpSampling2D((1,2))(process00)
    # process00 = conv12(process00) #17
    # process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    #process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    # process00 = UpSampling2D((1,2))(process00)
    # process00 = conv13(process00) #17
    # process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    #process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

    up_merged = UpSampling2D( (1, 8) )(merged)	
    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process10, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model    

def network_frame04_veloonly( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)
    
    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    
    up_merged = UpSampling2D( (1, 8) )(merged)	
#    process01 = Concatenate(axis = -1)( [up_merged, process10] )
    process01 = Conv2D(128, (1, 9), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 5), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (1, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    outputs = [process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model
  
def network_frame04_2branchd_earlyvelo( input_shape=(6, 32, 1024, 1), nb_classes = 2, relu_type = 'relu' ):
    #(samples, time, rows, cols, channels)
    # input_img = Input( shape=input_shape )
    # x = input_img

    # 1 * 2 * 32 * 1024 *1
    input_img = Input( shape=input_shape )
    input00 = Lambda(lambda x: x[:,0,:,:,:], name = "i_divided_0")(input_img)
    input01 = Lambda(lambda x: x[:,1,:,:,:], name = "i_divided_1")(input_img)
    input02 = Lambda(lambda x: x[:,2,:,:,:], name = "i_divided_2")(input_img)
    input03 = Lambda(lambda x: x[:,3,:,:,:], name = "i_divided_3")(input_img)
    human_mask = Lambda(lambda x: x[:,4,:,:,:], name = "human_mask")(input_img)
    defect_mask = Lambda(lambda x: x[:,5,:,:,:], name = "defect_mask")(input_img)

    inputframe=Input(shape=(32,1024,1))
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(inputframe)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split02 = UpSampling2D( (1,2) )(encode_split02)
    encode_split02 = Conv2D(64, (1, 3), padding='same', activation='relu')(encode_split02)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)
    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    encode_split03 = Conv2D(64, (1, 5), padding='same', activation='relu')(encode_split03)
#    encode_split03 = UpSampling2D( (1, 4) )(encode_split03)
    
    out=Add()([encode_split01,encode_split02,encode_split03])
    
    feature_model2 = Model(inputs=inputframe, outputs=out)

    conv11 = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11b = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12 = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12b = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv13 = Conv2D(32, (1, 3), activation='relu', padding='same')
    conv13b = Conv2D(32,(1, 3), activation='relu',padding='same')
    conv14 = Conv2D(32, (1, 3), padding='same', activation='relu')
    conv14b = Conv2D(16, (1, 3), padding='same', activation='relu')
    conv11s = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv11bs = Conv2D(128, (1, 3), padding='same', activation='relu')
    conv12s = Conv2D(64, (1, 3), padding='same', activation='relu')
    conv12bs = Conv2D(64, (1, 3), padding='same', activation='relu')

    merged00 = feature_model2(input00)
    merged01 = feature_model2(input01)
    merged02 = feature_model2(input02)
    merged03 = feature_model2(input03)
    
    merged = Concatenate(axis = -1)( [ merged00, merged01, merged02, merged03])
    merged_temporal = Concatenate(axis = -1)([merged00, merged01, merged02, merged03])

    encode_split01 = Conv2D(64, (1,1), padding='same', activation='relu')(input03)#1024
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input03)
    encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split02 = Conv2D(128, (1, 1), padding='same', activation='relu')(encode)#512
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(128, (3, 3), padding='same', activation='relu')(encode)
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode_split03 = Conv2D(512, (1, 1), padding='same', activation='relu')(encode)#256
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(256, (3, 3), padding='same', activation='relu')(encode)
#    encode_split03 = Conv2D(64, (1, 1), padding='same', activation='relu')(encode)#256
    encode = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    encode = Conv2D(512, (3, 3), padding='same', activation='relu')(encode)
    
#    spatial = Add()([encode_split01,encode_split02,encode_split03])
    spatial = encode

    process00 = UpSampling2D( (1, 2) )(merged_temporal)
    process00 = conv11(process00) #16  
    process00 = conv11b(process00)
    process10 = UpSampling2D( (1, 2) )(spatial)#256
    process10 = Add()([encode_split03, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv11s(process10) #16    
    process10 = conv11bs(process10)
    
    process00 = UpSampling2D((1,2))(process00)
    process00 = conv12(process00) #17
    process00 = conv12b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split02, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv12s(process10)
    process10 = conv12bs(process10)

    process00 = UpSampling2D((1,2))(process00)
    process00 = conv13(process00) #17
    process00 = conv13b(process00)
    process10 = UpSampling2D((1,2))(process10)
    process10 = Add()([encode_split01, process10])
    process10 = Concatenate(axis=-1)([process10, process00])
    process10 = conv14(process10) #18
    process10 = conv14b(process10)
#    process10 = Conv2D(2, (1, 1), activation='softmax', padding='same', name = 'segment')(process10)

 #   up_merged = UpSampling2D( (1, 2) )(process00)	
    process01 = Concatenate(axis = -1)( [process00, process10] )
    process01 = Conv2D(128, (3, 3), padding='same', activation='relu')(process01) #16
    process01 = Conv2D(64, (3, 3), padding='same', activation='relu')(process01) #17
    process01 = Conv2D(32, (3, 3), activation='relu', padding='same')(process01) #18

    process01 = Conv2D(2, (1, 1), activation='linear', padding='same', name = 'velocity')(process01)

    dprocess01 = Multiply(name='dprocess')([process01, defect_mask])
    wprocess01 = Multiply(name='wvelocity')([dprocess01, human_mask])

    process00 = Concatenate(axis = -1 )([process10, process01])
    process00 = Conv2D(2, (3, 3), activation = 'softmax', padding = 'same', name = 'segment')(process00)
    
    outputs = [process00, process01, dprocess01, wprocess01]
    model = Model(inputs=input_img, outputs=outputs)
    return model