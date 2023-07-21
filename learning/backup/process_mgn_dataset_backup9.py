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
data_processed_path = os.path.join(mgn_dataset_main_path, "ellipsoid01_test_5")
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()
visualization = False
verbose = False
num_pts = 1024
num_query_pts = 4000
data_point_count = len(os.listdir(data_processed_path))

fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
# selected_primitive_names = ["6polygon04"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]
# print("excluded_objects:", excluded_objects)

"""
*** bad grasp_idxs:
ellipsoid01-p1: 45      12, 13 
ellipsoid01-p2: 22, 23, 44, 46   
"""

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):  ["sphere04", "ellipsoid03-p1", "ellipsoid02-p1"]
# for idx, file_name in enumerate(sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))):
for idx, file_name in enumerate(["ellipsoid01-p1"]):    # "ellipsoid01-p1", "ellipsoid01-p2" "6polygon04" "sphere04", "ellipsoid03-p1", "ellipsoid02-p1"
    object_name = os.path.splitext(file_name)[0]

    # data_processed_path = os.path.join(mgn_dataset_main_path, f"shinghei_data_{object_name}")
    # os.makedirs(data_processed_path, exist_ok=True)
    # data_point_count = len(os.listdir(data_processed_path))

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
    partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)
    
    # # if visualization:
    # #     pcds = []
    # #     for j, pc in enumerate(partial_pcs):
    # #         pcds.append(pcd_ize(pc, color=[0,0,0]).translate((0,0.05*j,0)))
    # #     open3d.visualization.draw_geometries(pcds)
        
        
    for k in range(7,100):    # 100 grasp poses 6 p1, 7p1, 19 p1, 24 p2
        
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
        

        # for i in [0,10,14,15,17,18,20,21,22,23,26,27,29,30,32,33,35,36,38,40,41,43,44,46,47,49]:
        # for i in [0,17,20,26,29,32,33,35,36,38,40,41,43,44,46,47,49]:
        for i in [0, 49]:
        # for i in range(20,50):     # 50 time steps. Takes ~0.40 mins to process
            force = forces[i]  
            full_pc = full_pcs[i]
            print("time step, force:", i, force)
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
            signed_distance_full_pc = trimesh.proximity.signed_distance(object_mesh, full_pc)

            ### Points belongs the object volume            
            query_points_volume = trimesh.sample.volume_mesh(object_mesh, round(num_query_pts*2))[:num_query_pts]
            occupancy_volume = np.ones(query_points_volume.shape[0])
            # print("query_points_volume.shape", query_points_volume.shape)


            ### Gaussian random points (outside object mesh)              
            query_points_outside, is_inside = sample_points_gaussian(object_mesh, round(num_query_pts), scales=[1.5]*3, tolerance=0.001) 
            occupancy_outside = np.zeros(num_query_pts)
            occupancy_outside[np.where(is_inside==True)[0]] = 1


            assert query_points_outside.shape[0] == num_query_pts and query_points_volume.shape[0] == num_query_pts
                                       
            


            all_query_points = np.concatenate((query_points_volume, query_points_outside), axis=0)
            all_query_points[..., 1] -= 1.0    # shift back to (0,0,0) origin
            all_occupancies = np.concatenate((occupancy_volume, occupancy_outside), axis=0)     

            
            
            processed_data = {"query_points": all_query_points, "occupancy": all_occupancies,                                     
                            "force": force, 
                            "young_modulus": np.log(young_modulus), 
                            "object_name": object_name, "grasp_idx": k}
            
       

            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
            data_point_count += 1   


            # pcd_query_inside = pcd_ize(query_points_volume, color=[0,1,0], vis=False) 
            # pcd_full = pcd_ize(full_pc, color=[0,0,0], vis=False)
            # pcd_query_outside = pcd_ize(query_points_outside[is_inside==False], color=[1,0,0], vis=False) 
            # open3d.visualization.draw_geometries([pcd_query_outside, pcd_query_inside.translate((0.07,0,0)), pcd_full])
              

            # pcd_gripper = pcd_ize(data["gripper_pc"], color=[0,0,0])
            # # # pcd_queries = pcd_ize(all_query_points[np.where(all_occupancies==1)[0]], color=[0,0,0], vis=False)
            # pcd_full = pcd_ize(full_pcs[i], color=[1,0,0])
            # # # # colors = np.array(scalar_to_rgb(stresses[i], colormap='jet'))[:,:3]
            # # # # pcd_full.colors = open3d.utility.Vector3dVector(colors)
            # # # pcd_partial = pcd_ize(partial_pcs[2] + np.array([0,1.0,0]), color=[0,1,0])
            # open3d.visualization.draw_geometries([pcd_full, pcd_gripper])
            
            
            # open3d.visualization.draw_geometries([pcd_gripper, pcd_full, pcd_partial.translate((0.00,0,0))])


            # break
        break     
    
    
    
print_color(f"Final time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")
    
    