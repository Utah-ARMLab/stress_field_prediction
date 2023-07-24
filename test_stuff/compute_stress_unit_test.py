import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import open3d
from copy import deepcopy
import timeit
import sys
import isaacgym
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, read_pickle_data, down_sampling, scalar_to_rgb
import trimesh
from utils.stress_utils import *
from utils.process_data_utils import get_gripper_point_cloud

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data_6polygon04"
os.makedirs(gripper_pc_recording_path, exist_ok=True)

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon04_data"

start_time = timeit.default_timer()
num_pts = 1024

grasp_idx_bounds = [20, 30]


for object_name in ["6polygon04"]:


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tet_indices = static_data["tet_indices"] # shape (num_tetrahedra, 4)
    tri_indices = static_data["tri_indices"]
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in range(50,51):
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break   #continue  
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)


            full_pc = data["object_particle_state"] 
            tet_stress = data["tet_stress"] # shape (num_tetrahedra,): Cauchy stress tensors (3,3) at each tetrahedron 
            force = data["force"]
            print("force:", force)
            grasp_pose = data["grasp_pose"]
            fingers_joint_angles = data["fingers_joint_angles"]
            
            fingers_joint_angles[0] += 0.005
            fingers_joint_angles[1] += 0.005

            if get_gripper_pc:
                gripper_pc = get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=num_pts)
                pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])
                get_gripper_pc = False

            start_compute_time = timeit.default_timer()
            all_stresses = compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0])
            all_stresses_log = np.log(all_stresses)   
            print(f"End compute time: {(timeit.default_timer() - start_compute_time):.3f} seconds")
            
            pcd = pcd_ize(full_pc, color=[0,0,0])
            colors = np.array(scalar_to_rgb(all_stresses_log, colormap='jet'))[:,:3]
            pcd.colors = open3d.utility.Vector3dVector(colors) 

            open3d.visualization.draw_geometries([pcd, pcd_gripper])           