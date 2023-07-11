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
adjacent_tetrahedrals_save_path = "/home/baothach/shape_servo_data/stress_field_prediction/adjacent_tetrahedrals"

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/data"
data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data"
os.makedirs(data_processed_path, exist_ok=True)

data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = True
num_pts = 1024
num_query_pts = 5

grasp_idx_bounds = [0, 1]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]

# for object_name in OBJECT_NAMES:
for object_name in ["rectangle"]:

    get_gripper_pc = True

    # Get adjacent tetrahdras of each vertex
    with open(os.path.join(adjacent_tetrahedrals_save_path, f"{object_name}.pickle"), 'rb') as handle:
        adjacent_tetrahedral_dict = pickle.load(handle)   

    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    partial_pcs = static_data["partial_pcs"][1:2]  # list of 8 point clouds from 8 different camera views
    partial_pcs = [down_sampling(pc, num_pts=num_pts) for pc in partial_pcs]

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

                augmented_partial_pcs = [np.hstack((pc, np.tile(np.array([force, young_modulus]), (pc.shape[0], 1))))
                                        for pc in partial_pcs]  # list of 8 arrays of shape (num_pts,5)

                # Combine everything together to get an augmented point cloud of shape (num_pts*2,5)
                combined_pcs = [np.concatenate((pc, augmented_gripper_pc), axis=0)
                                for pc in augmented_partial_pcs]  # list of 8 arrays of shape (num_pts + num_pts, 5)
            
            
            if visualization:

                all_stresses = compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, object_particle_state.shape[0])
                all_stresses_log = np.log(all_stresses)        

                pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])

                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                colors = np.array(scalar_to_rgb(all_stresses_log, colormap='jet'))[:,:3]
                pcd_full.colors = open3d.utility.Vector3dVector(colors)

                pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])

                open3d.visualization.draw_geometries([pcd_partial.translate((0.1,0,0)), pcd_full, pcd_gripper])


            # Points belongs the object volume
            full_pc = object_particle_state
            selected_idxs = np.random.randint(low=0, high=full_pc.shape[0], size=num_query_pts)
            for idx in selected_idxs:
                query_point = full_pc[idx]
                stress = compute_stress_each_vertex(tet_stress, adjacent_tetrahedral_dict, vertex_idx=idx)
                stress_log = np.log(stress)

                for combined_pc in combined_pcs:
                    # Save data
                    processed_data = {"combined_pc": combined_pc.transpose(1,0),
                                    "stress": stress, "stress_log": stress_log, 
                                    "occupancy": 1, 
                                    "query_point": query_point, "object_name": object_name}
                    
                    with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                        
                    data_point_count += 1

            # Random points (outside object mesh)   
            sampled_points, outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                            boundary_threshold=[0.02,-0.01], \
                                            num_pts=num_query_pts, scales=[1.5, 1.5, 1.5], vis=visualization, seed=None, verbose=False)      
            
            for idx in outside_mesh_idxs:
                query_point = sampled_points[idx]

                for combined_pc in combined_pcs:
                    # Save data
                    processed_data = {"combined_pc": combined_pc.transpose(1,0),
                                    "stress": 0, "stress_log": -4, 
                                    "occupancy": 0, 
                                    "query_point": query_point, "object_name": object_name}
                    
                    with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                        
                    data_point_count += 1        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        