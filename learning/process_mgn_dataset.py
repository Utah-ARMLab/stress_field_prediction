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
Process the filtered data (originally from Isabella) for training neural network. 
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"

mgn_dataset_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"
raw_data_path = os.path.join(mgn_dataset_main_path, "raw_pickle_data")
data_path = os.path.join(mgn_dataset_main_path, "filtered_data")
data_processed_path = os.path.join(mgn_dataset_main_path, "processed_data")
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()
visualization = True
verbose = False
num_pts = 1024
num_query_pts = 5
data_point_count = len(os.listdir(data_processed_path))

fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]
# print("excluded_objects:", excluded_objects)

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):
# for idx, file_name in enumerate(sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))):
for idx, file_name in enumerate(["sphere02"]):
    object_name = os.path.splitext(file_name)[0]

    print("======================")
    print_color(f"{object_name}, {idx}")
    print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")

    if not any([prim_name in object_name for prim_name in selected_primitive_names]):   # if object does NOT belong to any of the selected primitives.
        print_color(f"{object_name} is not processed (type 1)")
        continue
    if any([excluded_object in object_name for excluded_object in excluded_objects]):   # if object belongs to the excluded object list.
        print_color(f"{object_name} is not processed (type 2)")
        continue 
    
    real_object_name = object_name
    if "-p1" in object_name or "-p2" in object_name:
        real_object_name = real_object_name[:-3]  # Ignore the -p1 and -p2 part.

    ### Get partial point clouds
    with open(os.path.join(static_data_recording_path, f"{real_object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    partial_pcs = static_data["partial_pcs"]  # list of 8 point clouds from 8 different camera views
    partial_pcs = np.array([down_sampling(pc, num_pts=num_pts) for pc in partial_pcs])   # shape (8, num_pts, 3)
    partial_pcs[..., 2] += 1.0  # add 1.0 to each z value of each point cloud   (to match with Isabella's data)
    partial_pcs[:, :, [1, 2]] = partial_pcs[:, :, [2, 1]]   # swap y and z values
    
    # if visualization:
    #     pcds = []
    #     for j, pc in enumerate(partial_pcs):
    #         pcds.append(pcd_ize(pc, color=[0,0,0]).translate((0,0.05*j,0)))
    #     open3d.visualization.draw_geometries(pcds)
        
    for k in range(0,100):    # 100 grasp poses
        
        file_name = os.path.join(data_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            print_color(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
  
        ### Get data for all 50 time steps
        full_pcs = data["object_particle_states"] 
        stresses = data["stresses"]  
        forces = data["forces"]  
        gripper_pc = data["gripper_pc"]
        tet_indices = data["tet_indices"]
        young_modulus = data["young_modulus"]
        
        augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), (gripper_pc.shape[0], 1)))) # shape (num_pts,5)

        if visualization:    
            for i in range(49,50):
                pcd_full = pcd_ize(full_pcs[i], color=[0,0,0])
                colors = np.array(scalar_to_rgb(stresses[i], colormap='jet'))[:,:3]
                pcd_full.colors = open3d.utility.Vector3dVector(colors)

                pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])
                pcd_partial = pcd_ize(partial_pcs[0], color=[1,0,0])    # just visualize partial pc from one of the camera
                open3d.visualization.draw_geometries([pcd_full.translate((-0.05,0,-0.05)), pcd_gripper, pcd_partial])
                
                break
        
        # for i in range(0,50):     # 50 time steps. Takes ~0.40 mins to process
        #     force = forces[i]  
        #     full_pc = full_pcs[i]
             
        #     augmented_partial_pcs = [np.hstack((pc, np.tile(np.array([force, young_modulus]), (pc.shape[0], 1))))
        #                             for pc in partial_pcs]  # list of 8 arrays of shape (num_pts,5)           
                
        #     # Combine everything together to get an augmented point cloud of shape (num_pts*2,5)
        #     combined_pcs = [np.concatenate((pc, augmented_gripper_pc), axis=0)
        #                     for pc in augmented_partial_pcs]  # list of 8 arrays of shape (num_pts + num_pts, 5)

        #     # Points belongs the object volume            
        #     selected_idxs = np.random.randint(low=0, high=full_pc.shape[0], size=num_query_pts)
        #     for idx in selected_idxs:
        #         query_point = full_pc[idx]
        #         stress = stresses[i][idx]              

        #         for combined_pc in combined_pcs:
        #             # Save data
        #             processed_data = {"combined_pc": combined_pc.transpose(1,0),
        #                             "stress_log": stress, "occupancy": 1,                                    
        #                             "query_point": query_point, "object_name": object_name, "grasp_idx": k}
                    
        #             with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
        #                 pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                        
        #             data_point_count += 1
                
        #     # Random points (outside object mesh)  
        #     sampled_points = sample_points_bounding_box(trimesh.PointCloud(full_pc), round(num_query_pts*1.5), scales=[1.5, 1.5, 1.5]) 
        #     is_inside = is_inside_tet_mesh_vectorized(sampled_points, vertices=full_pc, tet_indices=tet_indices)
        #     outside_mesh_idxs = np.where(is_inside == False)[0]
  
        #     if visualization:
        #         pcds = []
        #         pcd_full = pcd_ize(full_pc, color=[0,0,0])
        #         for i, query in enumerate(sampled_points):
        #             query_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        #             if i in outside_mesh_idxs:
        #                 color = [1,0,0]
        #             else:
        #                 color = [0,0,1]
        #             query_sphere.paint_uniform_color(color)
        #             query_sphere.translate(tuple(query))    
        #             pcds.append(query_sphere)
        #         open3d.visualization.draw_geometries(pcds + [pcd_full])  
                 
            
        #     outside_mesh_idxs = outside_mesh_idxs[:num_query_pts] # only select num_query_pts queries
        #     for idx in outside_mesh_idxs:
        #         query_point = sampled_points[idx]

        #         for combined_pc in combined_pcs:
        #             # Save data
        #             processed_data = {"combined_pc": combined_pc.transpose(1,0),
        #                             "stress_log": -4, "occupancy": 0,                                    
        #                             "query_point": query_point, "object_name": object_name, "grasp_idx": k}
                    
        #             with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
        #                 pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                        
        #             data_point_count += 1                   
            
                            
        #     # break   
        
        
        
        break     
    
    
    
print_color(f"Final time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")
    
    