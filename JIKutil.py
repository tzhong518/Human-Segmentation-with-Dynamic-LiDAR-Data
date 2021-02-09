import numpy as np
import random
import math
import scipy.spatial.distance
import csv
import pandas as pd


def getRandomPosition(r, theta):

	rand_r = random.randrange(r)
	rand_theta = random.randrange(theta)
	rand_x = rand_r * math.cos(math.radians(rand_theta))
	rand_y = rand_r * math.sin(math.radians(rand_theta))

	return rand_x, rand_y

def getRandomPosition_rec(x,y):
	while True:
		rand_x = random.randrange(x)
		rand_x = rand_x - x/2
		rand_y = random.randrange(y)
		if math.sqrt(rand_x*rand_x + rand_y*rand_y) > 500:
			break
	return rand_x, rand_y

def getRandomRotation(degree):
	rand_r = random.randrange(degree)
	Pointing_vector = [math.cos(math.radians(degree)),math.sin(math.radians(degree))]

	return rand_r, Pointing_vector

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def distance_line_point(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

def xyz2sph(xyz):
    sph = np.zeros((xyz.shape))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
    sph[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy))
    sph[:,1] = np.arctan2(xyz[:,1], xyz[:,0])

    return sph

def xy2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def closest_node(node, nodes):
    return nodes[scipy.spatial.distance.cdist([node], nodes).argmin()]

def convert_radian(deg):
    if deg < 180:
        return math.radians(deg)
    if deg >= 180:
        return math.radians(deg-360)


def Get_position_list_of_lidar(text_name):
    text = open(text_name, 'r')
    reader = csv.reader(text)
    text_list = []

    for each_line in reader:
        char_start = 11
        content_num = 1
        line_list = []
        line_list.append(each_line[0][0:10])
        for each_char in range(11,len(each_line[0])):
            if each_line[0][each_char:each_char+1] == ' ':
                line_list.append(each_line[0][char_start:each_char])
                char_start = each_char+1
                content_num += 1
            elif content_num == 4:
                line_list.append(each_line[0][char_start:])
                break
        text_list.append(line_list)
    refined_list = []
    for count_line in range(len(text_list)):
        if text_list[count_line][1] == '"odom"':
            refined_list.append(text_list[count_line])

    return refined_list

def Get_human_position_under_distance(floor, lidar_location, max_human_number, distance_min, distance_max):
    human_location_list = []
    ll_x = lidar_location[0]
    ll_y = lidar_location[1]
    ll_z = lidar_location[2]
    distance_min = 1000*distance_min
    distance_max = 1000*distance_max
    while True:
        picked_human_location = random.sample( list(floor), 1 )
        determine = math.sqrt( (picked_human_location[0][0]-ll_x)**2 + (picked_human_location[0][1]-ll_y)**2 + (picked_human_location[0][2]-ll_z)**2  )

        if (determine >= distance_min) and (determine <= distance_max):
            human_location_list.append(picked_human_location)
        
        if len(human_location_list) >= max_human_number:
            break
    return human_location_list

def get_angle_from_xy( x, y ):
    return np.degrees(np.arctan2(y, x))

def find_closest_point_index( array, point ):
    tmp = array[:, 0:1] - point
    return (tmp[:,0]**2 + tmp[:,1]**2).argmin()

def choices_consdiering_frame_length(trajectory_txt_list, k, frame_length):
    choiced_list = []
    while True:
        picked_list = random.choice(trajectory_txt_list)
        if len(picked_list) >= frame_length:
            choiced_list.append(picked_list)
        if len(choiced_list) == k:
            break
    return choiced_list

def choice_consdiering_frame_length(trajectory_txt_list, frame_length):
    while True:
        picked_list = random.choice(trajectory_txt_list)
        if len(picked_list) >= frame_length:
            break

    return picked_list

#def confirm_trajectory(picked_trajectory_txt, frame_length, Random_LiDAR_start_location_ID, picked_LiDAR_location):
#    f

def choices_consdiering_frame_length_under_critic(LiDAR_location_by_map, trajectory_txt_list, k, frame_length, MAP_parallel_displacement, Random_LiDAR_start_location_ID):
    choiced_list = []
    trajectory_count_array = np.zeros(k)
    trj_count = 0
    while True:
        picked_list = random.choice(trajectory_txt_list)
        length_criteria_txt = open(picked_list, 'r')
        length_criteria_pd_txt = pd.read_csv(length_criteria_txt)
        length_criteria_txt.close()

        if len(length_criteria_pd_txt) >= frame_length:
            append_checker = True
            tmp_trajectory_txt = open(picked_list, 'r')
            tmp_trajectory_pd = pd.read_csv(tmp_trajectory_txt)
            tmp_trajectory_array = tmp_trajectory_pd.values
            tmp_x = 1000 * tmp_trajectory_array[:,0] #+ MAP_parallel_displacement
            tmp_y = 1000 * tmp_trajectory_array[:,1]

            for LiDAR_location_count in range(frame_length):
                    picked_LiDAR_location = LiDAR_location_by_map[Random_LiDAR_start_location_ID+LiDAR_location_count]
                    LiDAR_location = 1000*np.array([picked_LiDAR_location[1],
                                                    picked_LiDAR_location[2]])
                    det_array = np.sqrt( (tmp_x-LiDAR_location[0])**2 + (tmp_y-LiDAR_location[1])**2 )
                        
                    if (det_array<500).any():
                        append_checker = False

            if append_checker:
                choiced_list.append(picked_list)
                trajectory_count_array[trj_count] = random.randrange(len(tmp_trajectory_pd)-frame_length+1)
                trj_count += 1
        if len(choiced_list) == k:
            break

    return choiced_list, trajectory_count_array

def Find_SC( SC_LIST, human_spec ):
    float_SC_LIST = np.asarray(SC_LIST, dtype=np.float32)
    ID_LIST = float_SC_LIST[:,0]
    SC = float_SC_LIST[np.where(ID_LIST == float(human_spec))][0, 1] * 2
    v_SC = float(SC) / 1.05
    return v_SC

def Get_rotation_matrix( MTV_MAT, timestamp ):
    mtv_mat_np = MTV_MAT.values
    time_np = mtv_mat_np[:,0]
    time_list = np.abs(time_np - timestamp)
    min_index = np.argmin(time_list)
    proper_mat = mtv_mat_np[min_index, 1:]
    rotation_matrix = proper_mat.reshape( 4,4 )
    return rotation_matrix

def Get_cmd_vel( CMD_VEL, timestamp ):
    cmd_vel_np = CMD_VEL.values
    time_np = cmd_vel_np[:,0]
    time_list = np.abs(time_np - timestamp)
    min_index = np.argmin(time_list)
    proper_mat = cmd_vel_np[min_index, :]
    return proper_mat

def Cal_back_vel( depth_array, tf_matrix, LiDAR_t, LiDAR_r ):
    back_vel_map = np.zeros((depth_array.shape[0], depth_array.shape[1], 2))
    for h in range(depth_array.shape[0]):
        for w in range(depth_array.shape[1]):
            if depth_array[h,w] > 0:
                base_x = (LiDAR_t + depth_array[h,w]*np.cos(LiDAR_r))
                base_y = (depth_array[h,w]*np.sin(LiDAR_r))
                velo_vel_array = np.dot(tf_matrix, np.array([ [base_x], [base_y], [0], [1] ]))
                back_vel_map[h,w,0] = velo_vel_array[0,0]
                back_vel_map[h,w,1] = velo_vel_array[1,0]
    return back_vel_map

def Cal_human_loc_vel( h_x, h_y, h_dx, h_dy, rotation_matrix ):
    map_location = np.array([ [h_x], [h_y], [0], [1] ])
    vel_location = np.dot(rotation_matrix, map_location)
    map_velocity = np.array([ [h_dx], [h_dy], [0], [1] ])
    vel_velocity = np.dot(rotation_matrix, map_velocity)
    human_loc_x = vel_location[0,0]
    human_loc_y = vel_location[1,0]
    human_vel_x = vel_velocity[0,0]
    human_vel_y = vel_velocity[1,0]
    return human_loc_x, human_loc_y, human_vel_x, human_vel_y
