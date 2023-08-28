
import os
import numpy as np
import pickle5 as pickle
import timeit
import sys
import argparse
import re
sys.path.append("../../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data, print_color
from utils.stress_utils import *
from utils.point_cloud_utils import transform_point_cloud
from utils.mesh_utils import sample_points_from_tet_mesh
from utils.constants import OBJECT_NAMES
sys.path.append("../")
from model import StressNet2, PointCloudEncoder
import open3d
""" 
Process evaluation data collected by Bao.
"""

def inverse_4x4_homogeneous_matrix(matrix):
    # Extract the upper-left 3x3 submatrix
    upper_left = matrix[:3, :3]

    # Calculate the inverse of the upper-left 3x3 submatrix
    upper_left_inv = np.linalg.inv(upper_left)

    # Extract the translation part (rightmost 3x1 column)
    translation = matrix[:3, 3]

    # Calculate the new translation using the inverse of the upper-left matrix
    new_translation = -np.dot(upper_left_inv, translation)

    # Construct the inverse matrix
    inverse_matrix = np.zeros_like(matrix)
    inverse_matrix[:3, :3] = upper_left_inv
    inverse_matrix[:3, 3] = new_translation
    inverse_matrix[3, 3] = 1

    return inverse_matrix

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"

data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/evaluate"


selected_objects = ["box01"]    #["mustard_bottle", "strawberry02", "lemon02"] "box01"


start_time = timeit.default_timer() 
visualization = False
process_gripper_only = False
save_gripper_data = True

num_pts = 1024
num_query_pts = 50000

grasp_idx_bounds = [0, 100]     # [0, 100]

device = torch.device("cuda")
# model = StressNet2(num_channels=5).to(device)
model = StressNet2(num_channels=5, pc_encoder_type=PointCloudEncoder).to(device)
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon/epoch 100"))
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon_open_gripper/epoch 193"))
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/weights/all_objects_occ/epoch 50"))
model.eval()


for object_name in selected_objects:    # 1,2,3,4,5,6,7,8

    prim_name = re.search(r'(\D+)', object_name).group(1)
    data_recording_path = os.path.join(data_main_path, f"all_{prim_name}_data")

    if not process_gripper_only:
        data_processed_path = os.path.join(data_main_path,  f"processed/processed_data_{object_name}")       
        os.makedirs(data_processed_path, exist_ok=True)
        data_point_count = len(os.listdir(data_processed_path))

    gripper_pc_recording_path = os.path.join(data_main_path,  f"processed/open_gripper_data_{object_name}") 
    os.makedirs(gripper_pc_recording_path, exist_ok=True)


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tri_indices = static_data["tri_indices"]
    tet_indices = static_data["tet_indices"]
    homo_mats = static_data["homo_mats"]
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    undeformed_full_pc = static_data["undeformed_full_pc"]
    undeformed_tet_stress = static_data["undeformed_tet_stress"]
    transformed_partial_pcs = static_data["transformed_partial_pcs"]  # shape (8, num_pts, 3)
    
    partial_pc = transformed_partial_pcs[0:1]
    # query = sample_points_bounding_box(trimesh.PointCloud(partial_pc.squeeze()), num_query_pts, scales=[3]*3)  # shape (num_query_pts,3)
    query = sample_points_bounding_box(trimesh.PointCloud(undeformed_full_pc), num_query_pts, scales=[1.5]*3)  # shape (num_query_pts,3)
    query = transform_point_cloud(query, homo_mats[0])
    query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
    query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)

    undeformed_object_mesh = trimesh.Trimesh(vertices=undeformed_full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))  # reconstruct the surface mesh.

    signed_distances = trimesh.proximity.signed_distance(undeformed_object_mesh, transform_point_cloud(query, inverse_4x4_homogeneous_matrix(homo_mats[0])))
    undeformed_gt_occupancy = (signed_distances >= 0.0).astype(int)
    # undeformed_gt_occupancy = is_inside_tet_mesh_vectorized(query, vertices=undeformed_full_pc, tet_indices=np.array(tet_indices).reshape(-1,4)).astype(int)

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in [0]:
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force_idx} started")
            
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
            


                
            if get_gripper_pc:
                gripper_file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{0}.pickle")
                                        
                if not os.path.isfile(gripper_file_name):
                    break 
                
                with open(gripper_file_name, 'rb') as handle:
                    gripper_data = pickle.load(handle)
                
                fingers_joint_angles = gripper_data["force_fingers_joint_angles"]   
                fingers_joint_angles = fingers_joint_angles[::-1]             
                fingers_joint_angles[0] += 0.005
                fingers_joint_angles[1] += 0.005

                # gripper_pc = get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=num_pts)
                gripper_pc = get_gripper_point_cloud(grasp_pose, [0.04,0.04], num_pts=num_pts, franka_gripper_mesh_main_path="../../graspsampling-py-defgraspsim")
                
                if save_gripper_data:
                    transformed_gripper_pcs = []
                    for i in range(8):
                        transformed_gripper_pcs.append(transform_point_cloud(gripper_pc, homo_mats[i])[np.newaxis, :]) 
                    transformed_gripper_pcs = np.concatenate(tuple(transformed_gripper_pcs), axis=0)     


                    gripper_data = {"gripper_pc": gripper_pc, "transformed_gripper_pcs": transformed_gripper_pcs}
                    save_path = os.path.join(gripper_pc_recording_path, f"{object_name}_grasp_{grasp_idx}.pickle")
                    write_pickle_data(gripper_data, save_path)
                
                get_gripper_pc = False
            
            
            if visualization:

                # pcd_partial = pcd_ize(partial_pcs[0], color=[0,0,0])
                pcd_full = pcd_ize(object_particle_state, color=[1,0,0])
                pcd_gripper = pcd_ize(gripper_pc, color=[0,1,0])

                # open3d.visualization.draw_geometries([pcd_partial, pcd_full, pcd_gripper])
                open3d.visualization.draw_geometries([pcd_full, pcd_gripper])


            ### DNN stuff
            
            augmented_partial_pcs = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus/1e4]]), 
                                                    (1, partial_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)    
            augmented_gripper_pc_open = np.concatenate([transformed_gripper_pcs[0:1], np.tile(np.array([[0, 0]]), 
                                                    (1, transformed_gripper_pcs[0:1].shape[1], 1))], axis=2)   # shape (8, num_pts, 5)   
            combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc_open), axis=1)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)

            stress, occupancy = model(combined_pc_tensor, query_tensor)

            pred_stress = stress.squeeze().cpu().detach().numpy()
            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.5)[0]
            recon_full_pc = query[occupied_idxs]
            recon_stress_field = pred_stress[occupied_idxs]            



            augmented_partial_pcs = np.concatenate([partial_pc, np.tile(np.array([[0, young_modulus/1e4]]), 
                                                    (1, partial_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)      
            combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc_open), axis=1)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)

            stress, occupancy = model(combined_pc_tensor, query_tensor)

            undeformed_pred_stress = stress.squeeze().cpu().detach().numpy()
            undeformed_pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(undeformed_pred_occupancy >= 0.5)[0]
            undeformed_recon_full_pc = query[occupied_idxs]


            # pcd_gripper = pcd_ize(transformed_gripper_pcs[0:1].squeeze(), color=[0,1,0])
            # pcd_recon = pcd_ize(recon_full_pc, color=[0,0,0])
            # pcd_undeformed_recon = pcd_ize(undeformed_recon_full_pc, color=[0,0,1]) 
            # pcd_undeformed_transformed = pcd_ize(transform_point_cloud(undeformed_full_pc, homo_mats[0]), color=[1,0,0])
            # pcd_gt_transformed = pcd_ize(transform_point_cloud(object_particle_state, homo_mats[0]), color=[0,0,1])
            # open3d.visualization.draw_geometries([pcd_recon.translate((-0.07,0,0)), pcd_undeformed_recon.translate((-0.14,0,0)),
            #                                       pcd_undeformed_transformed, pcd_gt_transformed.translate((0.07,0,0)),
            #                                       pcd_gripper])


            if process_gripper_only:
                break

            full_pc = object_particle_state            
            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))  # reconstruct the surface mesh.
            all_stresses = compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0])    # np.log needs fixing, e.g. np.log(0.0) undefined 
            all_stresses = np.where(all_stresses > 0, np.log(all_stresses), -4)

            undeformed_all_stresses = compute_all_stresses(undeformed_tet_stress, adjacent_tetrahedral_dict, undeformed_full_pc.shape[0])    # np.log needs fixing, e.g. np.log(0.0) undefined 
            undeformed_all_stresses = np.where(undeformed_all_stresses > 0, np.log(undeformed_all_stresses), -4)
            
            signed_distances = trimesh.proximity.signed_distance(object_mesh, transform_point_cloud(query, inverse_4x4_homogeneous_matrix(homo_mats[0])))
            gt_occupancy = (signed_distances >= 0.0).astype(int)
            # gt_occupancy = is_inside_tet_mesh_vectorized(query, vertices=full_pc, tet_indices=np.array(tet_indices).reshape(-1,4)).astype(int)
            # break
            
            processed_data = {"recon_full_pc": recon_full_pc, "recon_stress_field": recon_stress_field,
                            "gt_full_pc": full_pc, "gt_stress_field": all_stresses,      
                            "undeformed_recon_full_pc": undeformed_recon_full_pc,  
                            "gt_occupancy": gt_occupancy, "undeformed_gt_occupancy": undeformed_gt_occupancy,
                            "pred_occupancy": pred_occupancy, "undeformed_pred_occupancy": undeformed_pred_occupancy,
                            "undeformed_full_pc": undeformed_full_pc, "undeformed_stress_field": undeformed_all_stresses,
                            "query": query,                              
                            "force": force, "young_modulus": young_modulus,                             
                            "object_name": object_name, "grasp_idx": grasp_idx}
                                        
            
            with open(os.path.join(data_processed_path, f"processed sample {data_point_count}.pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            data_point_count += 1



            # pcd_query_inside = pcd_ize(query_points_volume, color=[0,1,0], vis=False) 
            # pcd_full = pcd_ize(full_pc, color=[0,0,0], vis=False)
            # pcd_query_outside = pcd_ize(query_points_outside, color=[1,0,0], vis=False) 
            # open3d.visualization.draw_geometries([pcd_query_outside, pcd_query_inside.translate((0.07,0,0)), pcd_full])                


            # pcd_gripper = pcd_ize(gripper_pc, color=[0,1,0])
            # force_fingers_joint_angles = data["force_fingers_joint_angles"]   
            # force_fingers_joint_angles = force_fingers_joint_angles[::-1]         
            # force_fingers_joint_angles[0] += 0.005
            # force_fingers_joint_angles[1] += 0.005
            # force_gripper_pc = get_gripper_point_cloud(grasp_pose, force_fingers_joint_angles, num_pts=num_pts)            
            # # partial_pcs = static_data["partial_pcs"][0]#.reshape(-1,3)
            # # pcd_partial = pcd_ize(partial_pcs, color=[0,1,0])
            # pcd_gripper_force = pcd_ize(force_gripper_pc, color=[1,0,0])
            
            # pcd = pcd_ize(full_pc, color=[0,0,0])
            # open3d.visualization.draw_geometries([pcd, pcd_gripper, pcd_gripper_force]) 


        #     break
        # break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        