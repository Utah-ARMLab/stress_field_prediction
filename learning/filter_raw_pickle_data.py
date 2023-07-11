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
mgn_dataset_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"
raw_data_path = os.path.join(mgn_dataset_main_path, "raw_pickle_data")
filtered_data_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
os.makedirs(filtered_data_path, exist_ok=True)

start_time = timeit.default_timer()
visualization = False
verbose = False
num_pts = 1024

fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]
# print("excluded_objects:", excluded_objects)

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):
for idx, file_name in enumerate(sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))):
# for idx, file_name in enumerate(["6polygon04"]):
    object_name = os.path.splitext(file_name)[0]

    if not any([prim_name in object_name for prim_name in selected_primitive_names]):   # if object does NOT belong to any of the selected primitives.
        print(f"{object_name} is not processed (type 1)")
        continue
    # if any([excluded_object in object_name for excluded_object in excluded_objects]):   # if object belongs to the excluded object list.
    #     print(f"{object_name} is not processed (type 2)")
    #     continue

    print("======================")
    print(object_name, idx)
    print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")

    for k in range(0,100):    # 100 grasp poses
        
        file_name = os.path.join(raw_data_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
  
        ### Get data for all 50 time steps
        all_object_gripper_combined_particle_states = data["world_pos"]  # shape (50, num_particles, 3)
        all_stresses = data["stress"]  # shape (50, num_particles)
        all_forces = data["force"]  # shape (50,)
        young_modulus = max(data["node_mod"])  # shape (num_particles, 3) ---MAX-->  scalar
        node_type = data["node_type"]  # shape (num_particles,)
        tet_indices = data["cells"]  # shape (num_particles, 4)
        
        ### Select only the object portion of everything
        first_object_index = next((i for i, num in enumerate(node_type) if num > 1), None)  # all particles from this index to the end, belong to the deformable object. Other particles belong to the gripper.
        # print("first_object_index:", first_object_index)
        tet_indices = tet_indices[np.where(np.min(tet_indices, axis=1) >= first_object_index)[0]] - first_object_index
        all_stresses = all_stresses[:, first_object_index:]  # select only stress values that belong to the object

        object_full_pcs = all_object_gripper_combined_particle_states[:, first_object_index:, :]  # shape (50, num_obj_particles, 3)
        gripper_full_pc = all_object_gripper_combined_particle_states[0, :first_object_index, :]  # gripper's point cloud when it makes contact with the object, but doesn't deform the object yet. 
        downsampled_gripper_pc = down_sampling(gripper_full_pc, num_pts=num_pts)       
        
        filter_data = {"object_particle_states": object_full_pcs, "gripper_pc": downsampled_gripper_pc, "tet_indices": tet_indices,
                       "stresses": all_stresses, "forces": all_forces, "young_modulus": young_modulus}
        with open(os.path.join(filtered_data_path, f"{object_name}_grasp_{k}.pickle"), 'wb') as handle:
            pickle.dump(filter_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        if visualization:    
            for i in range(49,50):     # 50 time steps
                # object_gripper_combined_particle_state = all_object_gripper_combined_particle_states[i]
                stress = all_stresses[i]
                force = all_forces[i]
                
                print("stress.shape, object_full_pcs[i].shape", stress.shape, object_full_pcs[i].shape)
                pcd_full = pcd_ize(object_full_pcs[i], color=[0,0,0])
                colors = np.array(scalar_to_rgb(stress, colormap='jet'))[:,:3]
                pcd_full.colors = open3d.utility.Vector3dVector(colors)

                # pcd_gripper = pcd_ize(downsampled_gripper_pc, color=[0,0,0])
                # open3d.visualization.draw_geometries([pcd_full, pcd_gripper])

                # print("object_full_pc.shape, gripper_pc.shape:", object_full_pc.shape, gripper_pc.shape)
                # print("force:", force)
                

                
                # pcd_full = pcd_ize(object_full_pc, color=[0,0,0])
                # pcd_gripper = pcd_ize(gripper_pc, color=[1,0,0])
                # pcd_test = pcd_ize(sampled_pc, color=[0,1,0])
                # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # # open3d.visualization.draw_geometries([pcd_full, pcd_test, coor.translate((0,1,0))])
                # open3d.visualization.draw_geometries([pcd_full, pcd_test, pcd_gripper])
                
            
            
                            
                break   
        
        
        
        # break     
    
    
    

    
    