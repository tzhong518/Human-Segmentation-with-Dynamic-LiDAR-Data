#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K

import cv2
import h5py
import numpy as np
import glob
import random
import math
import xml.etree.ElementTree as ET

import imagedegrade.np as degrade


def getNearetValue( list, value ):
	ind = ( np.abs( np.array( list ) - value ) ).argmin()
	return list[ind]



def median( src ):
	dst = src
	shape = src.shape
	while( len(np.where(dst==0)[0]) > 0 ):
	
		dst0 = dst
		for h in range(shape[0]):
			h0 = ( h-1 if h-1 >= 0 else 0 )
			h1 = ( h+2 if h+2 <= shape[0] else shape[0] )
			for w in range(shape[1]):
				w0 = ( w-1 if w-1 >= 0 else 0 )
				w1 = ( w+2 if w+2 <= shape[1] else shape[1] )
				
				if( dst0[h,w] == 0 ):
					d = dst0[h0:h1, w0:w1]
					d = d.ravel()
					d = d[d>0]
					if( len(d) > 0 ):
						d = list(d)
						d.sort(reverse=True)
						d0 = d.pop( len(d) // 2 )
						if( len(d) > 1 ):
							d1 = getNearetValue( d, d0 )
							dst[h,w] = (float(d0)+float(d1))/2.0
						else:
							dst[h,w] = d0

	return dst

def trans_depth_label(depth, label):
	depth = np.single(depth)
	label = np.single(label)
	
	depth = np.roll( depth, int(90.0 / ( 360.0 / 1650.0 )), axis=1 )
	label = np.roll( label, int(90.0 / ( 360.0 / 1650.0 )), axis=1 )

	ind0 = int(45.0 / (360.0/1660.0) )
	ind1 = int((360.0-45.0) / (360.0/1660.0) )

	depth = depth[:,ind0:ind1]
	label = label[:,ind0:ind1]

	depth = median( depth )
	
	return depth, label

def trans_depth(depth):
	depth = np.single(depth)
	
	depth = np.roll( depth, int(90.0 / ( 360.0 / 1650.0 )), axis=1 )

	ind0 = int(45.0 / (360.0/1660.0) )
	ind1 = int((360.0-45.0) / (360.0/1660.0) )

	depth = depth[:,ind0:ind1]

	depth = median( depth )
	
	return depth



def load_rand( dir, nb_data, nb_labels = 2 ):
	root = '/mnt/workspace/Depth_Gen/Data_Generator/Network/dataset/'
	files = glob.glob( root+dir+'/*.h5' )
	files = random.sample( files, nb_data )
	
	X = []
	Y = []
	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 1000 # 10[m] -> 1
		
		label = h5file['/label'].value

		ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
		for h in range(32):
			for w in range(1024):
				if label[h][w] > 0:
					ly[h][w][1] = 1
				else:
					ly[h][w][0] = 1	

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
			ly = K.permute_dimensions( ly, [2, 0, 1] )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
			
		X.append( lx )
		Y.append( ly )

	return np.asarray(X), np.asarray(Y)


def load3D( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files):
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				lv = np.single(h5file['/velocity'].value)
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) )
				xml_file = xml_directory + file[-16:-3] + '.xml'
				tree = ET.parse(xml_file)
				tvel = tree.findall('filename/LiDAR_vel_translation')
				rvel = tree.findall('filename/LiDAR_vel_rotation')
				loc = np.asarray([float(tvel[0].text), float(rvel[0].text)], dtype=np.float32)
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
			
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_loc.append( loc )
	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	L = np.asarray( seq_loc, dtype=np.float32 )
	return X, Y, V, L

def load_frame( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'/*.h5' ))

	seq_depth = []
	seq_labels = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files):
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )

				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
			
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )

	return X, Y

def load_formix_frame( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))

	seq_depth = []
	seq_labels = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files):
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )

				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
			
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )

	return X, Y

def load_real_for_ver02( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '../Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files):
				defect_mask = (lx > 0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )

		depth.append( mask )
		depth.append( defect_mask )
		
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	return X, Y

def load_real_for_ver02_mid( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = 'E:/data.tar/For_shousan/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files)-2:
				defect_mask = (lx > 0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
				count_label += 1
			elif count_label==len(files):
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )

		depth.append( mask )
		depth.append( defect_mask )
		
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	return X, Y

def load_demo_16( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)-nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		files = all_files[seq_count:seq_count+frame_rate]

		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 500
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
		depth.append( lx )
		depth.append( lx )
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )

	X = np.asarray( seq_depth, dtype=np.float32 )
	return X


def load_demo_01( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)-nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		files = all_files[seq_count:seq_count+frame_rate]

		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 500
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )

	X = np.asarray( seq_depth, dtype=np.float32 )
	return X


def load_for_ver02( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '../Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	#all_files=sorted(glob.glob('E:/data.tar/For_shousan/Data/Test_h5file/*.h5'))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r') #read HDF5 file
			lx = np.single(h5file['/depth'].value) #save depth(32,1024), just like accessing dictionary with key-value set. single is the data type
			lx = lx / 1000
			if count_label == len(files): 
				defect_mask = (lx > 0)#bool,(32,1024),True if depth value>0 (lx>0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) ) #(32,1024,1)
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) ) #(32,1024,2)
				lv = np.single(h5file['/velocity'].value) / 1000
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) ) #(32,1024,2)
				for h in range(32): #assign the segment label to ly
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0) #human label, bool, (32,1024)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) ) #(32,1024,1)
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )#(32,1024,1)
			depth.append( lx ) #list, [Numpy array, Numpy array, Numpy array, Numpy array]

		depth.append( mask )
		depth.append( defect_mask )
		
		lv = lv * defect_mask #(32,1024,2)
		array_depth = np.asarray( depth ) #list to numpy array (6,32,1024,1)
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_human_vel.append( lv * mask ) #list,(32,1024,2)

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	HV = np.asarray( seq_human_vel, dtype=np.float32 )
	return X, Y, V, HV

def load_for_ver04( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '../Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	#all_files=sorted(glob.glob('E:/data.tar/For_shousan/Data/Test_h5file/*.h5'))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int((nb_frame/frame_rate)*int(len(all_files)/nb_frame))

	for seq_count in range(seq_length):
		depth = []
		labels = []
		#frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = all_files[seq_count*frame_rate:seq_count*frame_rate+frame_rate]
		#files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r') #read HDF5 file
			lx = np.single(h5file['/depth'].value) #save depth(32,1024), just like accessing dictionary with key-value set. single is the data type
			lx = lx / 1000
			if count_label == len(files): 
				defect_mask = (lx > 0)#bool,(32,1024),True if depth value>0 (lx>0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) ) #(32,1024,1)
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) ) #(32,1024,2)
				lv = np.single(h5file['/velocity'].value) / 1000
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) ) #(32,1024,2)
				for h in range(32): #assign the segment label to ly
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0) #human label, bool, (32,1024)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) ) #(32,1024,1)
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )#(32,1024,1)
			depth.append( lx ) #list, [Numpy array, Numpy array, Numpy array, Numpy array]

		depth.append( mask )
		depth.append( defect_mask )
		
		lv = lv * defect_mask #(32,1024,2)
		array_depth = np.asarray( depth ) #list to numpy array (6,32,1024,1)
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_human_vel.append( lv * mask ) #list,(32,1024,2)

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	HV = np.asarray( seq_human_vel, dtype=np.float32 )
	return X, Y, V, HV

def load_for_49( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '../Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	#all_files=sorted(glob.glob('E:/data.tar/For_shousan/Data/Test_h5file/*.h5'))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = 29

	for seq_count in range(seq_length):
		depth = []
		labels = []
		#frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = all_files[seq_count:seq_count+frame_rate]
		#files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r') #read HDF5 file
			lx = np.single(h5file['/depth'].value) #save depth(32,1024), just like accessing dictionary with key-value set. single is the data type
			lx = lx / 1000
			if count_label == len(files): 
				defect_mask = (lx > 0)#bool,(32,1024),True if depth value>0 (lx>0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) ) #(32,1024,1)
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) ) #(32,1024,2)
				lv = np.single(h5file['/velocity'].value) / 1000
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) ) #(32,1024,2)
				for h in range(32): #assign the segment label to ly
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0) #human label, bool, (32,1024)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) ) #(32,1024,1)
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )#(32,1024,1)
			depth.append( lx ) #list, [Numpy array, Numpy array, Numpy array, Numpy array]

		depth.append( mask )
		depth.append( defect_mask )
		
		lv = lv * defect_mask #(32,1024,2)
		array_depth = np.asarray( depth ) #list to numpy array (6,32,1024,1)
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_human_vel.append( lv * mask ) #list,(32,1024,2)

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	HV = np.asarray( seq_human_vel, dtype=np.float32 )
	return X, Y, V, HV
    
def load_for_ver02_mid( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '../Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	#all_files=sorted(glob.glob('E:/data.tar/For_shousan/Data/Test_h5file/*.h5'))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r') #read HDF5 file
			lx = np.single(h5file['/depth'].value) #save depth(32,1024), just like accessing dictionary with key-value set. single is the data type
			lx = lx / 1000
			if count_label == len(files)-2: 
				defect_mask = (lx > 0)#bool,(32,1024),True if depth value>0 (lx>0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) ) #(32,1024,1)
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) ) #(32,1024,2)
				lv = np.single(h5file['/velocity'].value) / 1000
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) ) #(32,1024,2)
				for h in range(32): #assign the segment label to ly
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) ) #(32,1024,1)
				count_label += 1
			elif count_label==len(files):
				count_label=1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )#(32,1024,1)
			depth.append( lx ) #list, [Numpy array, Numpy array, Numpy array, Numpy array]

		depth.append( mask )
		depth.append( defect_mask )
		
		lv = lv * defect_mask #(32,1024,2)
		array_depth = np.asarray( depth ) #list to numpy array (6,32,1024,1)
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_human_vel.append( lv * mask ) #list,(32,1024,2)

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	HV = np.asarray( seq_human_vel, dtype=np.float32 )
	return X, Y, V, HV

def load_for_ver03( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_human_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			lxyz = np.single(h5file['/xyz'].value)
			lxyz = lxyz / 1000
			lxaxis = lxyz[:, :, 0]
			lyaxis = lxyz[:, :, 1]
			if count_label == len(files):
				defect_mask = (lx > 0)
				defect_mask = defect_mask.reshape( (defect_mask.shape[0], defect_mask.shape[1], 1) )
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				lv = np.single(h5file['/velocity'].value) / 1000
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) )
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				mask = (ly[:,:,1] > 0)
				mask = mask.reshape( (mask.shape[0], mask.shape[1], 1 ) )
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
			depth.append( lxaxis.reshape((lxyz.shape[0], lxyz.shape[1], -1 )) )
			depth.append( lyaxis.reshape((lxyz.shape[0], lxyz.shape[1], -1 )) )

		depth.append( mask )
		depth.append( defect_mask )
		lv = lv * defect_mask
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_human_vel.append( lv * mask )

	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	HV = np.asarray( seq_human_vel, dtype=np.float32 )
	return X, Y, V, HV



def load_loc( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_loc = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[-frame_rate:]
		count_label = 1
		for file in files:
			if count_label == len(files):
				xml_file = xml_directory + file[-16:-3] + '.xml'
				tree = ET.parse(xml_file)
				tvel = tree.findall('filename/LiDAR_vel_translation')
				rvel = tree.findall('filename/LiDAR_vel_rotation')
				loc = np.asarray([float(tvel[0].text), float(rvel[0].text)], dtype=np.float32)
				count_label = 1
			else:
				count_label += 1
		seq_loc.append( loc )
	L = np.asarray( seq_loc, dtype=np.float32 )
	return L


def load_sequential( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'_h5file/*.h5' ))
	xml_directory = root + dir + '_xml/'

	seq_depth = []
	seq_labels = []
	seq_vel = []
	seq_loc = []
	seq_length = int(len(all_files) - frame_rate)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files
		files = frame_files[seq_count:seq_count+frame_rate]
		count_label = 1
		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			if count_label == len(files):
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				lv = np.single(h5file['/velocity'].value)
				lv = lv.reshape( (lv.shape[0], lv.shape[1], 2 ) )
				xml_file = xml_directory + file[-16:-3] + '.xml'
				tree = ET.parse(xml_file)
				tvel = tree.findall('filename/LiDAR_vel_translation')
				rvel = tree.findall('filename/LiDAR_vel_rotation')
				loc = np.asarray([float(tvel[0].text), float(rvel[0].text)], dtype=np.float32)
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
				count_label = 1
			else:
				count_label += 1
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
			
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		seq_labels.append( ly )
		seq_vel.append( lv )
		seq_loc.append( loc )
	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	V = np.asarray( seq_vel, dtype=np.float32 )
	L = np.asarray( seq_loc, dtype=np.float32 )
	return X, Y, V, L


'''
def load3D( dir, nb_labels = 2,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'/*.h5' ))

	seq_depth = []
	seq_labels = []
	seq_length = int(len(all_files)/nb_frame)

	for seq_count in range(seq_length):
		depth = []
		labels = []
		frame_files = all_files[seq_count*nb_frame:seq_count*nb_frame+nb_frame]
		files = frame_files[nb_frame-frame_rate-1:]
		for count_seq, file in enumerate(files):
			h5file = h5py.File(file, 'r')
			if (count_seq+1) == len(files):
				label = np.single(h5file['/label'].value)
				ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
				for h in range(32):
					for w in range(1024):
						if label[h][w] > 0:
							ly[h][w][1] = 1
						else:
							ly[h][w][0] = 1
			else:
				lx = np.single(h5file['/depth'].value)
				lx = lx / 1000
				lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
				depth.append( lx )
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
		labels.append( ly )
		array_labels = np.asarray( labels )
		seq_labels.append( ly )
	#print(seq_depth)
	#print(seq_labels)
	X = np.asarray( seq_depth, dtype=np.float32 )
	Y = np.asarray( seq_labels, dtype=np.float32 )
	return X, Y
'''
def load3D_real( dir,nb_frame=32, frame_rate = 16 ):
	root = '/mnt/datagrid1/LiDAR_DataGen/SequenceGenerator/SequenceVer01/Training_space/Data/'
	all_files = sorted(glob.glob( root+dir+'/*.h5' ))

	seq_depth = []
	seq_length = int(len(all_files)-nb_frame)
	for seq_count in range(seq_length):
		depth = []
		frame_files = all_files[seq_count:seq_count+nb_frame]
		#print(len(frame_files))
		files = []
		skip_count = int(nb_frame/frame_rate)
		for frame_count in range(frame_rate):
			#print(skip_count*frame_count+skip_count - 1)
			files.append(frame_files[nb_frame - frame_rate + frame_count])

		for file in files:
			h5file = h5py.File(file, 'r')
			lx = np.single(h5file['/depth'].value)
			lx = lx / 1000
			lx = lx.reshape( (lx.shape[0], lx.shape[1], -1 ) )
			depth.append( lx )
		array_depth = np.asarray( depth )
		seq_depth.append( array_depth )
	X = np.asarray( seq_depth, dtype=np.float32 )
	return X




def load( dir, nb_labels = 2, mirror=False ):
	root = '/mnt/workspace/Depth_Gen/Data_Generator/Network/Validation/'
	files = glob.glob( root+dir+'/*.h5' )
	X = []
	Y = []

	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 1000 # 1[m] -> 1
		
		label = h5file['/label'].value

		ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
		for h in range(32):
			for w in range(1024):
				if label[h][w] > 0:
					ly[h][w][1] = 1
				else:
					ly[h][w][0] = 1	

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
			ly = K.permute_dimensions( ly, [2, 0, 1] )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
		
		X.append( lx )
		Y.append( ly )
		if( mirror ):
			X.append( np.fliplr( lx ) )
			Y.append( np.fliplr( ly ) )
	return np.asarray(X), np.asarray(Y)

def load_train( dir, nb_labels = 2, mirror=False ):
	root = '/mnt/workspace/Depth_Gen/Data_Generator/Network/dataset/'
	files = sorted(glob.glob( root+dir+'/*.h5' ))
	X = []
	Y = []

	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 1000 # 1[m] -> 1
		
		label = h5file['/label'].value

		ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
		for h in range(32):
			for w in range(1024):
				if label[h][w] > 0:
					ly[h][w][1] = 1
				else:
					ly[h][w][0] = 1	

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
			ly = K.permute_dimensions( ly, [2, 0, 1] )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
		
		X.append( lx )
		Y.append( ly )
		if( mirror ):
			X.append( np.fliplr( lx ) )
			Y.append( np.fliplr( ly ) )
	return np.asarray(X), np.asarray(Y)


def load_test( dir, nb_labels = 2, mirror=False ):
	root = '/mnt/workspace/Depth_Gen/Data_Generator/Network/Test/'
	files = glob.glob( root+dir+'/*.h5' )
	X = []
	Y = []

	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 1000 # 1[m] -> 1
		
		label = h5file['/label'].value

		ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
		for h in range(32):
			for w in range(1024):
				if label[h][w] > 0:
					ly[h][w][1] = 1
				else:
					ly[h][w][0] = 1	

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
			ly = K.permute_dimensions( ly, [2, 0, 1] )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
		
		X.append( lx )
		Y.append( ly )
		if( mirror ):
			X.append( np.fliplr( lx ) )
			Y.append( np.fliplr( ly ) )
	return np.asarray(X), np.asarray(Y)

root = '/mnt/workspace/human_mesh_vtk/Network/Data/'
def loadreal( dir, nb_labels = 2, mirror=False ):
	files = sorted(glob.glob( root+dir+'/*.h5' ))
	X = []
	Y = []

	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 1000 # 1[m] -> 1
		
		label = h5file['/label'].value

		ly = np.zeros( (label.shape[0], label.shape[1], nb_labels ) )
		for h in range(32):
			for w in range(1024):
				if label[h][w][1] > 0:
					ly[h][w][1] = 1
				else:
					ly[h][w][0] = 1	

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
			ly = K.permute_dimensions( ly, [2, 0, 1] )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
		
		X.append( lx )
		Y.append( ly )
		if( mirror ):
			X.append( np.fliplr( lx ) )
			Y.append( np.fliplr( ly ) )
	return np.asarray(X), np.asarray(Y)

def load_depth( dir, trans=False ):
	files = glob.glob( root+dir+'/*.h5' )
	files = sorted(files)
	X = []
	for file in files:
		h5file = h5py.File(file,'r')
		
		lx = np.single(h5file['/depth'].value)
		lx = lx / 10000 # 10[m] -> 1
		if( trans ):
			lx = trans_depth( lx )

		if( K.image_data_format() == 'channels_first' ):
			lx = lx.reshape( (1, lx.shape[0], lx.shape[1] ) )
		else:
			lx = lx.reshape( (lx.shape[0], lx.shape[1], 1 ) )
			
		X.append( lx )
	return np.asarray(X)

def save_depth_predict( fileformat, depth, predict ):
	s_depth = depth.shape
	s_predict = predict.shape
	depth *= 10000 # 1 -> 10[m]
	depth = np.uint16( depth )
	predict = np.single( predict )
	for i in range(min(s_depth[0],s_predict[0])):
		filename = fileformat.format(n=i)
		h5file = h5py.File(root+filename, 'w')
		h5file.create_dataset( '/depth', data=depth[i,] )
		h5file.create_dataset( '/predict', data=predict[i,] )
		h5file.flush()
		h5file.close()

def pixelwise_categorical_accuracy(y_true, y_pred):
	if( K.image_data_format() == 'channels_first' ):
		axis = 1
	else:
		axis = 3

	eva = K.equal( K.argmax(y_true, axis=axis), K.argmax(y_pred, axis=axis) )
	eva = K.cast( eva, K.floatx() )
	eva = K.mean( eva )
	return eva



def croppingwidth( X, Y, num=32 ):
	width = X.shape[2]
	nwidth = (width//num)*num
	if( width > nwidth ):
		st = (width-nwidth)//2
		X = X[:,:,st:st+nwidth,:]
		Y = Y[:,:,st:st+nwidth,:]

	return X, Y

def croppingwidth1( X, num=32 ):
	width = X.shape[2]
	nwidth = (width//num)*num
	if( width > nwidth ):
		st = (width-nwidth)//2
		X = X[:,:,st:st+nwidth,:]

	return X


def get_weight(img,sigma,mu):
	f = (1/(math.sqrt(2*math.pi*sigma*sigma)))*np.exp(-((img-mu)**2)/(2*(sigma**2)))
	return f

def blur_depth_map(lx):
	lx = lx.reshape( (lx.shape[1], lx.shape[2], lx.shape[3]) )
	t_lx = np.empty((lx.shape[0], lx.shape[1], 3))
	t_lx[:,:,0] = lx.copy().reshape((lx.shape[0], lx.shape[1]))
	t_lx[:,:,1] = lx.copy().reshape((lx.shape[0], lx.shape[1]))
	t_lx[:,:,2] = lx.copy().reshape((lx.shape[0], lx.shape[1]))

	mask = t_lx > 0
	#ones = np.ones((mask.shape))
	ones = 1 * mask
	defect_mask = ~mask.copy()
	while True:
		blur_lx = t_lx.copy()

		blur_lx = cv2.blur(blur_lx, (5,5))

		print(np.unique(blur_lx - t_lx))
		blur_ones = 1*(blur_lx > 0)
		print(len(np.where(blur_ones==0)))

		if len(np.where(blur_ones==0)) == 0:
			break
		else:
			t_lx = blur_lx

	blur_image = t_lx * mask + blur_lx * defect_mask
	return blur_image[:,:,0].reshape((blur_image.shape[0], blur_image.shape[1],1))

def get_weight_sigmoid(img, center):
	sigmoid = 1 / ( 1 + np.exp(-10 * (img - center)) )
	return sigmoid
