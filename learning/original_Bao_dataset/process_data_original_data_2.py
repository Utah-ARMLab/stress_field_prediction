import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse

import isaacgym
sys.path.append("../../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES

""" 
Process data collected by Bao
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data"
os.makedirs(gripper_pc_recording_path, exist_ok=True)

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/data"
data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_2"
os.makedirs(data_processed_path, exist_ok=True)

data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = False
num_pts = 1024
# num_query_pts = 4000

grasp_idx_bounds = [0, 100]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]

# for object_name in OBJECT_NAMES:
for object_name in ["6polygon04"]:

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = False#True
        
        for force in force_levels:
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force}.pickle")
            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break 
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)

            (tet_indices, tet_stress) = data["tet"]
            tet_indices = np.array(tet_indices).reshape(-1,4)
            (tri_indices, tri_parents, tri_normals) = data["tri"]
            
            # object_name = data["object_name"]        
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

                ### Get partial point clouds
                with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
                    static_data = pickle.load(handle)
                partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)

                pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])
                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                pcd_gripper = pcd_ize(gripper_pc, color=[0,1,0])

                # open3d.visualization.draw_geometries([pcd_partial, pcd_full, pcd_gripper])
                open3d.visualization.draw_geometries([pcd_full, pcd_gripper])


            full_pc = object_particle_state
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
            signed_distance_full_pc = trimesh.proximity.signed_distance(object_mesh, full_pc)
            # print("full_pc.shape:", full_pc.shape)

            ### Points belongs to the object volume   
            query_points_volume = full_pc
            occupancy_volume = np.ones(query_points_volume.shape[0])
            signed_distances_volume = signed_distance_full_pc   #trimesh.proximity.signed_distance(object_mesh, query_points_volume)  
            num_query_pts = query_points_volume.shape[0]         

            ### Random points (both outside and inside object mesh)   
            outside_mesh_idxs = None
            while(outside_mesh_idxs is None or outside_mesh_idxs.shape[0] < num_query_pts):                
                query_points_random, signed_distances_random, \
                outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                    boundary_threshold=[0.02, min(signed_distance_full_pc)], \
                                    num_pts=round(num_query_pts*1.3), scales=[1.5]*3, vis=False, seed=None, verbose=False)  
                
                
            outside_mesh_idxs = outside_mesh_idxs[:num_query_pts]
            query_points_random = query_points_random[outside_mesh_idxs]
            occupancy_random = np.zeros(num_query_pts)
            signed_distances_random = signed_distances_random[outside_mesh_idxs]

            
            all_query_points = np.concatenate((query_points_volume, query_points_random), axis=0)
            all_occupancies = np.concatenate((occupancy_volume, occupancy_random), axis=0)     
            all_signed_distances = np.concatenate((signed_distances_volume, signed_distances_random), axis=0)        
            
            
            processed_data = {"query_points": all_query_points, "occupancy": all_occupancies,                                     
                            "signed_distance": all_signed_distances, "force": force, 
                            "young_modulus": np.log(young_modulus), 
                            "object_name": object_name, "grasp_idx": grasp_idx}
                                        

            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
            data_point_count += 1    


            # break
        # break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        