import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse

sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb, print_color
from utils.stress_utils import *


""" 
Filter raw data collected by Isabella. Remove unnecessary information.
"""

dgn_dataset_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset"
raw_data_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/raw_pickle_data"
filtered_data_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
os.makedirs(filtered_data_path, exist_ok=True)

visualization = True
fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
for idx, object_name in enumerate([f"ellipsoid0{j}" for j in [1,2,3,4]]):
# for idx, file_name in enumerate(sorted(os.listdir(raw_data_path))):


    for k in range(100):    # 100 grasp poses
        
        print("======================")
        print(object_name, idx)
        
        # if not any([fruit_name in object_name for fruit_name in ["potato","eggplant","cucumber"]]):
        #     break
        
        mesh = trimesh.load(os.path.join(dgn_dataset_path, object_name, f"{object_name}.stl"))


        
        file_name = os.path.join(raw_data_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            break
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
  
        # Get data for all 50 time steps
        all_object_gripper_combined_particle_states = data["world_pos"]  # shape (50, num_particles, 3)
        all_stresses = data["stress"]  # shape (50, num_particles)
        all_forces = data["force"]  # shape (50, num_particles)
        young_modulus = data["node_mod"]  # shape (num_particles, 3)
        node_type = data["node_type"]  # shape (num_particles,)
        tet_indices = data["cells"]  # shape (num_particles, 4)
        print("all_object_gripper_combined_particle_states.shape:", all_object_gripper_combined_particle_states.shape)
        print("tet_indices.shape:", tet_indices.shape)
        print(max(tet_indices.flatten()), min(tet_indices.flatten()))
        first_object_index = next((i for i, num in enumerate(node_type) if num > 1), None)  # all particles from this index to the end, belong to the deformable object. Other particles belong to the gripper.
        print("first_object_index:", first_object_index)
        tet_indices = tet_indices[np.where(np.min(tet_indices, axis=1) >= first_object_index)[0]] - first_object_index
        

        sampled_pc = trimesh.sample.sample_surface_even(mesh, count=1024)[0]+ np.array([0,0,1.0])# + np.array([tfn[3],tfn[5],1.0])  #+ np.array([0,0,1.0])
        sampled_pc[:, [1, 2]] = sampled_pc[:, [2, 1]]
        if any([fruit_name in object_name for fruit_name in fruit_names]):
            print_color(object_name)
            sampled_pc[:,2] *= -1
                
        for i in range(1):     # 50 time steps
            object_gripper_combined_particle_state = all_object_gripper_combined_particle_states[i]
            stress = all_stresses[i]
            force = all_forces[i]
            
            object_full_pc = object_gripper_combined_particle_state[first_object_index:]
            gripper_pc = object_gripper_combined_particle_state[:first_object_index]

            
            pcd_full = pcd_ize(object_full_pc, color=[0,0,0])
            pcd_gripper = pcd_ize(gripper_pc, color=[1,0,0])
            pcd_test = pcd_ize(sampled_pc, color=[0,1,0])
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # open3d.visualization.draw_geometries([pcd_full, pcd_test, coor.translate((0,1,0))])
            open3d.visualization.draw_geometries([pcd_full, pcd_test, pcd_gripper])
            

        
                        
            break   
        
        
        
        break     
    
    
    
    
    
    