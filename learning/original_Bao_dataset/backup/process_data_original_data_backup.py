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
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES

""" 
Process data collected by Bao
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/data"
data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data"
os.makedirs(data_processed_path, exist_ok=True)

data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = True
num_pts = 1024
# num_query_pts = 5

grasp_idx_bounds = [0, 1]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]

# for object_name in OBJECT_NAMES:
for object_name in ["6polygon04"]:

    get_gripper_pc = True



    for i in range(*grasp_idx_bounds):        
        # print(f"{object_name} - grasp {i} started. Time passed: {timeit.default_timer() - start_time}")
        
        for force in force_levels:
            
            print(f"{object_name} - grasp {i} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{i}_force_{force}.pickle")
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

            

                
            if get_gripper_pc:
                gripper_pc = get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=num_pts)
                augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), (gripper_pc.shape[0], 1))))
                get_gripper_pc = False


            
            
            if visualization:

                ### Get partial point clouds
                with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
                    static_data = pickle.load(handle)
                partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)

                pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])
                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])

                open3d.visualization.draw_geometries([pcd_partial, pcd_full, pcd_gripper])


            num_query_pts = object_particle_state.shape[0]

            # Points belongs the object volume
            full_pc = object_particle_state
            occupancy = np.ones(full_pc.shape[0])
            signed_distances = np.zeros(full_pc.shape[0])

            # Random points (outside object mesh)   
            outside_mesh_idxs = None
            while(outside_mesh_idxs is None or outside_mesh_idxs.shape[0] < num_query_pts):
                sampled_points, signed_distances, \
                outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                    boundary_threshold=[0.02,-0.01], \
                                    num_pts=round(num_query_pts*1.7), scales=[1.2]*3, vis=visualization, seed=None, verbose=False)      
                
            outside_mesh_idxs = outside_mesh_idxs[:num_query_pts]
            query_points_outside = sampled_points[outside_mesh_idxs]
            occupancy_outside = np.zeros(num_query_pts)
            signed_distances_outside = signed_distances[outside_mesh_idxs]
            
            all_query_points = np.concatenate((full_pc, query_points_outside), axis=0)
            all_occupancies = np.concatenate((occupancy, occupancy_outside), axis=0)     
            all_signed_distances = np.concatenate((signed_distances, signed_distances_outside), axis=0)        
            
            
            processed_data = {"query_points": all_query_points, "occupancy": all_occupancies,                                     
                            "signed_distances": all_signed_distances, "force": force, 
                            "young_modulus": np.log(young_modulus), "object_name": object_name}
                                        

            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
            data_point_count += 1    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        