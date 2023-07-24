import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse

import isaacgym
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES

""" 
Process data collected by Bao.
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data_6polygon04"
os.makedirs(gripper_pc_recording_path, exist_ok=True)

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon04_data"
data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_6polygon04"
os.makedirs(data_processed_path, exist_ok=True)

data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = False
num_pts = 1024
# num_query_pts = 2000

grasp_idx_bounds = [0, 100]


# for object_name in OBJECT_NAMES:
for object_name in ["6polygon04"]:

    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tri_indices = static_data["tri_indices"]
    tet_indices = static_data["tet_indices"]
    # partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in range(0,61):
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break   #continue  
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)

                       
            tet_stress = data["tet_stress"]        
            young_modulus = int(float(data["young_modulus"]))
            object_particle_state = data["object_particle_state"]
            force = data["force"]
            grasp_pose = data["grasp_pose"]
            fingers_joint_angles = data["fingers_joint_angles"]
            
            fingers_joint_angles[0] += 0.005
            fingers_joint_angles[1] += 0.005

                
            if get_gripper_pc:
                gripper_pc = get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=num_pts)
                gripper_data = {"gripper_pc": gripper_pc}
                save_path = os.path.join(gripper_pc_recording_path, f"{object_name}_grasp_{grasp_idx}.pickle")
                write_pickle_data(gripper_data, save_path)
                
                get_gripper_pc = False
          
            
            if visualization:

                # pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])
                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                pcd_gripper = pcd_ize(gripper_pc, color=[0,1,0])

                # open3d.visualization.draw_geometries([pcd_partial, pcd_full, pcd_gripper])
                open3d.visualization.draw_geometries([pcd_full, pcd_gripper])


            full_pc = object_particle_state            
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
            # signed_distance_full_pc = trimesh.proximity.signed_distance(object_mesh, full_pc)
            # print("full_pc.shape:", full_pc.shape)


            ### Points belongs the object volume            
            query_points_volume = full_pc
            occupancy_volume = np.ones(query_points_volume.shape[0])
            num_query_pts = query_points_volume.shape[0]    


            ### Gaussian random points (outside object mesh)              
            query_points_outside, is_inside = sample_points_gaussian(object_mesh, round(num_query_pts), scales=[1.2]*3, tolerance=0.0005) 
            occupancy_outside = np.zeros(num_query_pts)
            occupancy_outside[np.where(is_inside==True)[0]] = 1

            # if query_points_outside.shape[0] != num_query_pts or query_points_volume.shape[0] != num_query_pts:
            #     print(query_points_outside.shape[0], query_points_volume.shape[0])

            assert query_points_outside.shape[0] == num_query_pts and query_points_volume.shape[0] == num_query_pts
                                       


            all_query_points = np.concatenate((query_points_volume, query_points_outside), axis=0)
            all_occupancies = np.concatenate((occupancy_volume, occupancy_outside), axis=0)    
            
            
            processed_data = {"query_points": all_query_points, "occupancy": all_occupancies,                                     
                            "force": force, 
                            "young_modulus": np.log(young_modulus), 
                            "object_name": object_name, "grasp_idx": grasp_idx}
                                        

            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
            data_point_count += 1    


            # break
        # break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        