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
data_processed_path = os.path.join(mgn_dataset_main_path, "shinghei_data_sdf")
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()
visualization = False
verbose = False
num_pts = 1024
# num_query_pts = 10000
data_point_count = len(os.listdir(data_processed_path))

fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
selected_primitive_names = ["ellipsoid"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]
# print("excluded_objects:", excluded_objects)

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):
# for idx, file_name in enumerate(sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))):
for idx, file_name in enumerate(["ellipsoid01-p2"]):
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
    tri_indices = static_data["tri_indices"]
    # partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)
    
    # # if visualization:
    # #     pcds = []
    # #     for j, pc in enumerate(partial_pcs):
    # #         pcds.append(pcd_ize(pc, color=[0,0,0]).translate((0,0.05*j,0)))
    # #     open3d.visualization.draw_geometries(pcds)
        
        
    for k in range(0,100):    # 100 grasp poses
        
        print(f"{object_name} - grasp {k} started. Time passed: {(timeit.default_timer() - start_time)/60:.2f}\n")
        
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
        tet_indices = data["tet_indices"]
        young_modulus = data["young_modulus"]
        

        if visualization:    
            for i in range(49,50):
                pcd_full = pcd_ize(full_pcs[i], color=[0,0,0])
                colors = np.array(scalar_to_rgb(stresses[i], colormap='jet'))[:,:3]
                pcd_full.colors = open3d.utility.Vector3dVector(colors)

                # pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])
                # pcd_partial = pcd_ize(partial_pcs[0], color=[1,0,0])    # just visualize partial pc from one of the camera
                # open3d.visualization.draw_geometries([pcd_full.translate((-0.05,0,-0.05)), pcd_gripper, pcd_partial])

                mesh = open3d.geometry.TriangleMesh()
                mesh.vertices = open3d.utility.Vector3dVector(full_pcs[i])
                mesh.triangles = open3d.utility.Vector3iVector(np.array(tri_indices).astype(np.int32))
                
                open3d.visualization.draw_geometries([mesh, pcd_full.translate((0.07,0,0))])
                
                break
        
        num_query_pts = full_pcs[0].shape[0]
        for i in range(0,50):     # 50 time steps. Takes ~0.40 mins to process
            force = forces[i]  
            full_pc = full_pcs[i]

            ### Points belongs the object volume            
            query_points = full_pc
            stress = stresses[i]
            occupancy = np.ones(stress.shape)
            
               
            ### Random points (outside object mesh)  
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
            signed_distance_full_pc = trimesh.proximity.signed_distance(object_mesh, full_pc)
            
            outside_mesh_idxs = None
            while(outside_mesh_idxs is None or outside_mesh_idxs.shape[0] < num_query_pts):
                # sampled_points = sample_points_bounding_box(trimesh.PointCloud(full_pc), round(num_query_pts*1.7), scales=[1.2]*3) 
                # is_inside = is_inside_tet_mesh_vectorized(sampled_points, vertices=full_pc, tet_indices=tet_indices)
                # outside_mesh_idxs = np.where(is_inside == False)[0]

                query_points_random, signed_distances_random, \
                outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                    boundary_threshold=[0.02, min(signed_distance_full_pc)], \
                                    num_pts=round(num_query_pts*1.5), scales=[1.5]*3, vis=False, seed=None, verbose=False)                 
                # print(f"num_outside/total: {outside_mesh_idxs.shape[0]}/{num_query_pts}")

                            
            outside_mesh_idxs = outside_mesh_idxs[:num_query_pts] # only select num_query_pts queries
            query_points_outside = query_points_random[outside_mesh_idxs]
            stress_outside = 0.0001 * np.ones(outside_mesh_idxs.shape[0])     #-4 * np.ones(outside_mesh_idxs.shape[0])
            occupancy_outside = np.zeros(stress_outside.shape)


            all_query_points = np.concatenate((query_points, query_points_outside), axis=0)
            all_stresses = np.concatenate((stress, stress_outside), axis=0)
            all_occupancies = np.concatenate((occupancy, occupancy_outside), axis=0)
            # print(all_query_points.shape, all_stresses.shape, all_occupancies.shape)
            

            processed_data = {"stress_log": all_stresses, "occupancy": all_occupancies,                                    
                            "query_points": all_query_points, "force": force, "young_modulus": young_modulus,
                            "object_name": object_name, "grasp_idx": k}            

            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
            data_point_count += 1   
                            
            # break   
        
        
        
        # break     
    
    
    
print_color(f"Final time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")
    
    