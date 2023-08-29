import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import sys
import argparse
import re

sys.path.append("../../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data, print_color, read_pickle_data
from utils.stress_utils import *
from utils.point_cloud_utils import transform_point_cloud
from utils.mesh_utils import point_cloud_to_mesh_mcubes, mise_voxel
from utils.constants import OBJECT_NAMES
import matplotlib.pyplot as plt
from copy import deepcopy

def kendall_tau_between_arrays(arr1, arr2):
    from scipy.stats import kendalltau

    # Rank the values in each array
    rank1 = np.argsort(np.argsort(arr1))
    rank2 = np.argsort(np.argsort(arr2))
    
    # print(arr1)
    # print(arr2)
    # print(rank1)
    # print(rank2)
    
    # Compute the Kendall Tau correlation between the rankings
    kendall_tau, _ = kendalltau(rank1, rank2)
    
    return kendall_tau

def chamfer_distance(pointcloud1, pointcloud2):
    from scipy.spatial import cKDTree

    # Ensure pointcloud1 and pointcloud2 are both 2D arrays with shape (num_points, 3)
    if pointcloud1.ndim == 1:
        pointcloud1 = pointcloud1.reshape(-1, 1)
    if pointcloud2.ndim == 1:
        pointcloud2 = pointcloud2.reshape(-1, 1)
    
    tree1 = cKDTree(pointcloud1)
    distances_1to2, _ = tree1.query(pointcloud2)
    
    tree2 = cKDTree(pointcloud2)
    distances_2to1, _ = tree2.query(pointcloud1)
    
    chamfer_dist = np.mean(distances_1to2) + np.mean(distances_2to1)
    
    return chamfer_dist



def performance_metric_ground_truth(gt_full_pc, gt_stress_field, undeformed_full_pc, undeformed_stress_field):
    deformation_delta = np.sum(np.abs(gt_full_pc - undeformed_full_pc))
    stress_delta = np.sum(gt_stress_field)

    return deformation_delta, stress_delta 


def performance_metric_predicted(recon_full_pc, recon_stress_field, undeformed_full_pc, undeformed_stress_field):
    deformation_delta = chamfer_distance(recon_full_pc, undeformed_full_pc)
    stress_delta = chamfer_distance(recon_stress_field, undeformed_stress_field)

    return deformation_delta, stress_delta 


def performance_metric_ground_truth_BCE(gt_occupancy, undeformed_gt_occupancy):

    import torch
    import torch.nn as nn

    y_true = gt_occupancy
    y_pred = undeformed_gt_occupancy

    # Convert NumPy arrays to PyTorch tensors
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    bce_loss = nn.BCELoss()


    deformation_delta = bce_loss(y_pred_tensor, y_true_tensor).item()
    stress_delta = None

    return deformation_delta, stress_delta 


def performance_metric_predicted_BCE(pred_occupancy, undeformed_pred_occupancy, recon_stress_field):

    import torch
    import torch.nn as nn

    y_true = pred_occupancy
    y_pred = undeformed_pred_occupancy
    # print(y_true)
    # print(y_pred)
    # Convert NumPy arrays to PyTorch tensors
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    bce_loss = nn.BCELoss()


    deformation_delta = bce_loss(y_pred_tensor, y_true_tensor).item()
    stress_delta = np.sum(recon_stress_field)

    return deformation_delta, stress_delta 

def performance_metric_ground_truth_euclidean(gt_occupancy, undeformed_gt_occupancy):

    # print(gt_occupancy)
    # print(undeformed_gt_occupancy)
    # print("==========================")

    deformation_delta = np.linalg.norm(gt_occupancy - undeformed_gt_occupancy)
    stress_delta = None

    return deformation_delta, stress_delta 


def performance_metric_predicted_euclidean(pred_occupancy, undeformed_pred_occupancy, recon_stress_field):
    pred_occupancy_threshold = (pred_occupancy >= 0.99).astype(int)
    undeformed_pred_occupancy_threshold = (undeformed_pred_occupancy >= 0.99).astype(int)
    
    # print(pred_occupancy_threshold)
    # print(undeformed_pred_occupancy_threshold)
    # print("==========================")
    
    deformation_delta = np.linalg.norm(pred_occupancy_threshold - undeformed_pred_occupancy_threshold)
    stress_delta = np.sum(recon_stress_field)

    return deformation_delta, stress_delta 

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/evaluate"


selected_objects =  ["mustard_bottle", "strawberry02", "lemon02"]    # ["mustard_bottle", "strawberry02", "lemon02"] "box01"


start_time = timeit.default_timer() 


grasp_idx_bounds = [0, 100]     # [0, 100]
temp_boolean = True


for object_name in selected_objects:    # 1,2,3,4,5,6,7,8

    print_color(f"object name: {object_name}")

    prim_name = re.search(r'(\D+)', object_name).group(1)

    data_processed_path = os.path.join(data_main_path,  f"processed/processed_data_{object_name}")       
    gripper_pc_recording_path = os.path.join(data_main_path,  f"processed/open_gripper_data_{object_name}") 


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tri_indices = static_data["tri_indices"]
    tet_indices = static_data["tet_indices"]
    homo_mats = static_data["homo_mats"]
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    undeformed_full_pc = static_data["undeformed_full_pc"]
    
    transformed_partial_pcs = static_data["transformed_partial_pcs"]  # shape (8, num_pts, 3)

    recon_deformation_deltas = []
    recon_stress_deltas = []
    gt_deformation_deltas = []
    gt_stress_deltas = []
    chamfer_distances = []

    for idx in range(0,100,2):
        file_name = os.path.join(data_processed_path, f"processed sample {idx}.pickle")
        if not os.path.isfile(file_name):
            # print_color(f"{file_name} not found")
            continue          

        processed_data = read_pickle_data(file_name)

        object_name = processed_data["object_name"]
        grasp_idx = processed_data["grasp_idx"]
        undeformed_stress_field = processed_data["undeformed_stress_field"]
        recon_full_pc = processed_data["recon_full_pc"]
        recon_stress_field = processed_data["recon_stress_field"]
        undeformed_recon_full_pc = processed_data["undeformed_recon_full_pc"]
        gt_full_pc = processed_data["gt_full_pc"]
        gt_stress_field = processed_data["gt_stress_field"]
        gt_occupancy = processed_data["gt_occupancy"]
        undeformed_gt_occupancy = processed_data["undeformed_gt_occupancy"]
        pred_occupancy = processed_data["pred_occupancy"]
        query = processed_data["query"]

        if temp_boolean:
            undeformed_pred_occupancy = processed_data["undeformed_pred_occupancy"]
            # temp_boolean = False

        gripper_pc = read_pickle_data(data_path=os.path.join(gripper_pc_recording_path, 
                                                             f"{object_name}_grasp_{grasp_idx}.pickle"))["transformed_gripper_pcs"][0:1]

        # recon_deformation_delta, recon_stress_delta = \
        #     performance_metric_predicted(recon_full_pc, recon_stress_field, undeformed_recon_full_pc, undeformed_stress_field)
        
        recon_deformation_delta, recon_stress_delta = performance_metric_predicted_euclidean(pred_occupancy, undeformed_pred_occupancy, recon_stress_field)
        recon_deformation_deltas.append(recon_deformation_delta)
        recon_stress_deltas.append(recon_stress_delta)
        
        # gt_deformation_delta, gt_stress_delta = performance_metric_ground_truth(gt_full_pc, gt_stress_field, undeformed_full_pc, undeformed_stress_field)
        gt_deformation_delta, gt_stress_delta = performance_metric_ground_truth_euclidean(gt_occupancy, undeformed_gt_occupancy)
        gt_deformation_deltas.append(gt_deformation_delta)
        gt_stress_deltas.append(gt_stress_delta)        

        # pcd_recon = pcd_ize(undeformed_recon_full_pc, color=[0,0,0], vis=True)

        # mesh = point_cloud_to_mesh_mcubes(undeformed_recon_full_pc, grid_size=[12]*3, use_mcubes_smooth=False,
        #                                   vis_voxel=False, vis_mesh=True)
        # mesh = mise_voxel(undeformed_recon_full_pc, initial_voxel_resolution=32, final_voxel_resolution=32, voxel_size=0.1)
        # # mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=2, iterations=1)
        # # mesh.show()

        # point_cloud = trimesh.points.PointCloud(transform_point_cloud(undeformed_full_pc, homo_mats[0])*10)
        # point_cloud.apply_translation([0,0,3])
        # scene = trimesh.Scene([mesh,point_cloud])
        # scene.show()

    #     pcd_partial = pcd_ize(transformed_partial_pcs[0], color=[0,0,1])
    #     pcd_recon = pcd_ize(undeformed_recon_full_pc, color=[0,0,0])
    #     pcd_undeformed_transformed = pcd_ize(transform_point_cloud(undeformed_full_pc, homo_mats[0]), color=[1,0,0])
    #     pcd_gt_transformed = pcd_ize(transform_point_cloud(gt_full_pc, homo_mats[0]), color=[0,0,1])
    #     # open3d.visualization.draw_geometries([pcd_recon.translate((0.07,0,0)), pcd_undeformed_transformed, pcd_gt_transformed.translate((0.07,0,0))])
        
    #     # open3d.visualization.draw_geometries([pcd_recon.translate((0.15,0,0)), pcd_undeformed_transformed])
    #     open3d.visualization.draw_geometries([pcd_partial.translate((-0.07,0,0)), pcd_undeformed_transformed, pcd_recon.translate((0.07,0,0))])
        
    #     chamfer_distances.append(chamfer_distance(recon_full_pc, transform_point_cloud(gt_full_pc, homo_mats[0])))


        # delta_vis = 0.1 
        # pcd_recon = pcd_ize(query[pred_occupancy >= 0.5], color=[0,0,0])
        # pcd_undeformed_recon = pcd_ize(query[undeformed_pred_occupancy >= 0.5], color=[1,0,0]).translate((delta_vis,0,0))
        # pcd_gt = pcd_ize(query[gt_occupancy >= 0.5], color=[0,1,0]).translate((delta_vis*2,0,0))
        # pcd_undeformed_gt = pcd_ize(query[undeformed_gt_occupancy >= 0.5], color=[0,0,1]).translate((delta_vis*3,0,0))        
        # open3d.visualization.draw_geometries([pcd_recon, pcd_undeformed_recon, pcd_gt, pcd_undeformed_gt])

        
        # break
        
    kendall_tau = kendall_tau_between_arrays(recon_deformation_deltas, gt_deformation_deltas)
    print("Kendall Tau deformation:", kendall_tau)
    # print(len(recon_deformation_deltas), len(gt_deformation_deltas))
    # a = np.arange(100)
    # b = np.arange(1,101)
    # b[99] = 0
    # test = kendall_tau_between_arrays(a, b)
    # print(test)
    
    # chamfer_distances = np.array(chamfer_distances) * 1000 / 2
    # # print(np.max(chamfer_distances), np.min(chamfer_distances), np.mean(chamfer_distances))
    # print(np.mean(chamfer_distances))
    
    # # correlation = np.corrcoef(recon_deformation_deltas, gt_deformation_deltas)[0, 1]
    # # print("Pearson Correlation Coefficient deformation:", correlation)
        
    # # kendall_tau = kendall_tau_between_arrays(recon_stress_deltas, gt_stress_deltas)
    # # print("Kendall Tau stress:", kendall_tau)
    
    # # correlation = np.corrcoef(recon_stress_deltas, gt_stress_deltas)[0, 1]
    # # print("Pearson Correlation Coefficient stress:", correlation)    
    
    # # # Normalize the arrays to the range [0, 1]
    # # normalized_recon = (recon_deformation_deltas - np.min(recon_deformation_deltas)) / (np.max(recon_deformation_deltas) - np.min(recon_deformation_deltas))
    # # normalized_gt = (gt_deformation_deltas - np.min(gt_deformation_deltas)) / (np.max(gt_deformation_deltas) - np.min(gt_deformation_deltas))

    # # # Plot the normalized arrays
    # # plt.plot(normalized_recon, label='Reconstructed')
    # # plt.plot(normalized_gt, label='Ground Truth')
    # # plt.xlabel('Index')
    # # plt.ylabel('Normalized Values')
    # # plt.title('Normalized Deformation Deltas')
    # # plt.legend()
    # # plt.show()
    






       
        
        
        
        
        
        
        
        
        
        
        