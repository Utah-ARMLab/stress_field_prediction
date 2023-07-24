import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import open3d
from copy import deepcopy
import timeit
import sys
import random
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, read_pickle_data, down_sampling
import trimesh

def upsample_point_cloud(point_cloud, k, num_upsamples):
    """
    Upsample a point cloud using k-nearest neighbor interpolation.

    Parameters:
        point_cloud (numpy array): Input point cloud of shape (N, 3) where N is the number of points.
        k (int): The number of nearest neighbors to use for interpolation.
        num_upsamples (int): The number of times to upsample the point cloud.

    Returns:
        numpy array: Upsampled point cloud of shape (M, 3) where M is the number of upsampled points.
    """
    upsampled_point_cloud = point_cloud.copy()
    
    for _ in range(num_upsamples):
        # Use k-nearest neighbors to find the indices of the k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(upsampled_point_cloud)
        _, indices = nbrs.kneighbors(upsampled_point_cloud)

        # Compute the interpolated points for each point using the k-nearest neighbors
        interpolated_points = np.mean(upsampled_point_cloud[indices], axis=1)

        # Concatenate the original point cloud with the interpolated points
        upsampled_point_cloud = np.concatenate((upsampled_point_cloud, interpolated_points), axis=0)
    
    return upsampled_point_cloud

def generate_random_weights(k):
    # Generate random weights for k points
    weights = np.random.rand(k, 4)
    # Normalize the weights so that they sum up to 1 for each point
    normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
    return normalized_weights

def compute_weighted_average(vertices, weights):
    # Compute the weighted average for each set of weights and vertices
    return np.einsum('ijk,ij->ik', vertices, weights)

def sample_points_from_mesh(mesh, k):
    num_tetrahedra = mesh.shape[0]
    vertices = mesh.reshape(num_tetrahedra, 4, 3)

    points = []

    for _ in range(k):
        # Generate random weights for all tetrahedra and points at once
        weights_list = generate_random_weights(k=1)

        # Compute the weighted average location for all points and vertices
        sampled_points = compute_weighted_average(vertices, weights_list)
        
        points.append(sampled_points)
        
    sampled_points = np.concatenate((points), axis=0)

    return sampled_points


def sample_points_from_mesh_old(mesh, k):
    num_tetrahedra = mesh.shape[0]
    vertices = mesh.reshape(num_tetrahedra, 4, 3)

    # Generate random weights for all tetrahedra and points at once
    weights_list = generate_random_weights(k)

    # Repeat vertices for k points to match the size of weights_list
    vertices_repeated = np.repeat(vertices, k, axis=0)

    # Compute the weighted average location for all points and vertices
    sampled_points = compute_weighted_average(vertices_repeated, weights_list)

    return sampled_points


static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data_6polygon04"
os.makedirs(gripper_pc_recording_path, exist_ok=True)

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon04_data"
data_processed_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_6polygon04"
os.makedirs(data_processed_path, exist_ok=True)

data_point_count = len(os.listdir(data_processed_path))
start_time = timeit.default_timer() 
visualization = False
num_pts = 1024
# num_query_pts = 2000

grasp_idx_bounds = [0, 1]


# for object_name in OBJECT_NAMES:
for object_name in ["6polygon04"]:


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    tet_indices = static_data["tet_indices"]
    tri_indices = static_data["tri_indices"]

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in range(0,1):
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break   #continue  
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)


            full_pc = data["object_particle_state"]            
            pcd = pcd_ize(full_pc, color=[0,0,0])
            
            start_time = timeit.default_timer()
            full_pc_upsampled = sample_points_from_mesh(full_pc[tet_indices], k = 2)
            # # full_pc_upsampled = down_sampling(full_pc_upsampled, num_pts=2000)
            # # full_pc_upsampled = np.concatenate((full_pc, full_pc_upsampled[:2000-full_pc.shape[0]]))
            # selected_idxs = np.random.choice(full_pc_upsampled.shape[0], size=4000-full_pc.shape[0], replace=False)
            # full_pc_upsampled = np.concatenate((full_pc, full_pc_upsampled[selected_idxs]))
                                
            
            print("Method 1 time: ", timeit.default_timer() - start_time)

            object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
            start_time = timeit.default_timer()          
            query_points_volume = trimesh.sample.volume_mesh(object_mesh, 2000)
            print("Method 2 time: ", timeit.default_timer() - start_time)
            
            
            print(full_pc.shape, full_pc_upsampled.shape)
            pcd_upsampled = pcd_ize(full_pc_upsampled.reshape(-1,3), color=[1,0,0])
            
            
            open3d.visualization.draw_geometries([pcd, pcd_upsampled.translate((0.08,0,0))]) 
