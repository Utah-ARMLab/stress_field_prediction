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
from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data, sample_points_from_tet_mesh, print_color
from utils.stress_utils import *
from utils.point_cloud_utils import transform_point_cloud
from utils.constants import OBJECT_NAMES

""" 
Process data collected by Bao.
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
# gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data_6polygon04"
# os.makedirs(gripper_pc_recording_path, exist_ok=True)

data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
data_recording_path = os.path.join(data_main_path, "all_6polygon_data")
# data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_6polygon04"
# os.makedirs(data_processed_path, exist_ok=True)

# data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = False
process_gripper_only = True
num_pts = 1024
num_query_pts = 2000

grasp_idx_bounds = [0, 100]


for object_name in [f"6polygon0{j}" for j in [4]]:

    if not process_gripper_only:
        data_processed_path = os.path.join(data_main_path,  f"processed_data_{object_name}")       
        os.makedirs(data_processed_path, exist_ok=True)
        data_point_count = len(os.listdir(data_processed_path))
    gripper_pc_recording_path = os.path.join(data_main_path,  f"gripper_data_{object_name}") 
    os.makedirs(gripper_pc_recording_path, exist_ok=True)


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tri_indices = static_data["tri_indices"]
    tet_indices = static_data["tet_indices"]
    homo_mats = static_data["homo_mats"]
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    # partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)
    

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in range(0,61):
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                # print_color(f"{file_name} not found")
                break   
            
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
                transformed_gripper_pcs = []
                for i in range(8):
                    transformed_gripper_pcs.append(transform_point_cloud(gripper_pc, homo_mats[i])[np.newaxis, :]) 
                transformed_gripper_pcs = np.concatenate(tuple(transformed_gripper_pcs), axis=0)     
                # print(transformed_gripper_pcs.shape)  

                gripper_data = {"gripper_pc": gripper_pc, "transformed_gripper_pcs": transformed_gripper_pcs}
                save_path = os.path.join(gripper_pc_recording_path, f"{object_name}_grasp_{grasp_idx}.pickle")
                write_pickle_data(gripper_data, save_path)
                
                get_gripper_pc = False
            
            if process_gripper_only:
                break
            
            if visualization:

                # pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])
                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                pcd_gripper = pcd_ize(gripper_pc, color=[0,1,0])

                # open3d.visualization.draw_geometries([pcd_partial, pcd_full, pcd_gripper])
                open3d.visualization.draw_geometries([pcd_full, pcd_gripper])


            full_pc = object_particle_state            
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))  # reconstruct the surface mesh.
            all_stresses = np.log(compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0]))    # np.log needs fixing, e.g. np.log(0.0) undefined 
            # all_stresses = np.random.uniform(size=full_pc.shape[0])

            # pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])
            # pcd = pcd_ize(full_pc, color=[0,0,0])
            # colors = np.array(scalar_to_rgb(all_stresses, colormap='jet'))[:,:3]
            # pcd.colors = open3d.utility.Vector3dVector(colors) 
            # open3d.visualization.draw_geometries([pcd, pcd_gripper]) 


            ### Points belongs the object volume - (positive samples)   
            if num_query_pts > full_pc.shape[0]:                  
                   
                full_pc_w_stress = np.concatenate((full_pc, all_stresses[:, np.newaxis]), axis=1)   # shape (num_pts, 4)

                num_pts_each_tetrahedron = int(np.ceil((num_query_pts-full_pc.shape[0]) / tet_indices.shape[0]))
                points_from_tet_mesh = sample_points_from_tet_mesh(full_pc_w_stress[tet_indices], k=num_pts_each_tetrahedron)   # sample points from the volumetric mesh
                selected_idxs = np.random.choice(points_from_tet_mesh.shape[0], size=num_query_pts-full_pc.shape[0], replace=False) # only select a few points sampled from the tet mesh.
                
                query_points_volume = np.concatenate((full_pc, points_from_tet_mesh[selected_idxs][:,:3])) # concat the object particles with these newly selected points.
                stress_volume = np.concatenate((full_pc_w_stress[:,3:], points_from_tet_mesh[selected_idxs][:,3:])).squeeze()   # stress at each query points

                # print("num_pts_each_tetrahedron:", num_pts_each_tetrahedron) 
                
                
            else:
                query_points_volume = full_pc[:num_query_pts]
            
            occupancy_volume = np.ones(query_points_volume.shape[0])   # occupancy at each query points


            ### Gaussian random points (outside object mesh) - (negative samples)              
            query_points_outside = sample_points_gaussian_3(object_mesh, round(num_query_pts), scales=[1.2]*3, tolerance=0.0005) 
            occupancy_outside = np.zeros(query_points_outside.shape[0])
            stress_outside = -4 * np.ones(query_points_outside.shape[0])


            assert query_points_outside.shape[0] == num_query_pts and query_points_volume.shape[0] == num_query_pts
                                       

            all_query_points = np.concatenate((query_points_volume, query_points_outside), axis=0)
            all_occupancies = np.concatenate((occupancy_volume, occupancy_outside), axis=0)    
            all_stresses = np.concatenate((stress_volume, stress_outside), axis=0) 


            
            processed_data = {"query_points": all_query_points, "occupancy": all_occupancies,                                     
                            "stress_log": all_stresses, "force": force, 
                            "young_modulus": np.log(young_modulus), 
                            "object_name": object_name, "grasp_idx": grasp_idx}
                                        
            
            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            data_point_count += 1



            # pcd_query_inside = pcd_ize(query_points_volume, color=[0,1,0], vis=False) 
            # pcd_full = pcd_ize(full_pc, color=[0,0,0], vis=False)
            # pcd_query_outside = pcd_ize(query_points_outside, color=[1,0,0], vis=False) 
            # open3d.visualization.draw_geometries([pcd_query_outside, pcd_query_inside.translate((0.07,0,0)), pcd_full])                


            # break
        # break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        