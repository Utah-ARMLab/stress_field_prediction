import sys
from utils.graspsampling_utilities import poses_wxyz_to_mats
from utils.hands import create_gripper
import trimesh
import torch
import open3d
import numpy as np

def get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=1024, gripper_name='panda'):
    gripper = create_gripper(gripper_name, configuration=fingers_joint_angles, 
                             franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=True)
    transformation_matrix = poses_wxyz_to_mats(grasp_pose)[0]
    mesh = gripper.mesh.copy()
    mesh.apply_transform(transformation_matrix)
    return trimesh.sample.sample_surface_even(mesh, count=num_pts)[0]

def get_stress_query_points(model, device, query_list, augmented_partial_pc, augmented_aps, batch_size):
    """ 
    query_list: np array, list of query points. Shape (num_qrs, 3). 
    augmented_partial_pc: partial pc with one-hot encoding. Shape (1024, 6).
    augmented_aps: attachment points with one-hot encoding. Shape (100, 6).
 
    Returns: np array, list of predicted stress scalars at each query points. Shape (num_qrs,).   
    """
    
    num_qrs = query_list.shape[0]
    queries_tensor = torch.from_numpy(query_list).float().to(device)  # shape (num_qrs, 3)
    combined_pc = np.concatenate((augmented_partial_pc, augmented_aps), axis=0)
    combined_pc_tensor = torch.from_numpy(combined_pc).float().permute(1,0).unsqueeze(0).repeat(num_qrs,1,1).to(device)    # shape (num_qrs,6,1024)

    with torch.no_grad():
        outputs = []
        for batch_combined_pc, batch_queries in zip(torch.split(combined_pc_tensor, batch_size), torch.split(queries_tensor, batch_size)):  # split into smaller batches to fit GPU memory
            outputs.append(model(batch_combined_pc, batch_queries)) # stress prediction


    return torch.cat(tuple(outputs), dim=0).squeeze().cpu().detach().numpy()

def get_stress_occupancy_query_points(model, device, query_list, augmented_partial_pc, augmented_aps, batch_size, occupancy_threshold):
    """ 
    query_list: np array, list of query points. Shape (num_qrs, 3). 
    augmented_partial_pc: partial pc with one-hot encoding. Shape (1024, 6).
    augmented_aps: attachment points with one-hot encoding. Shape (100, 6).
 
    Returns: (O1, O2, O3)
    O1: np array, list of predicted stress scalars at each query points. Shape (num_qrs,).   
    O2: np array, confidence score [0,1] of whether each query point belongs to the volume of the object. Shape (num_qrs,). 
    O3: indices of all query points that the model is confident to belong to the volume of the object (likelihood >= occupancy_threshold). 
    """
    
    num_qrs = query_list.shape[0]
    queries_tensor = torch.from_numpy(query_list).float().to(device)  # shape (num_qrs, 3)
    combined_pc = np.concatenate((augmented_partial_pc, augmented_aps), axis=0)
    combined_pc_tensor = torch.from_numpy(combined_pc).float().permute(1,0).unsqueeze(0).repeat(num_qrs,1,1).to(device)    # shape (num_qrs,6,1024)

    with torch.no_grad():
        stress_outputs = []
        occupancy_outputs = []
        for batch_combined_pc, batch_queries in zip(torch.split(combined_pc_tensor, batch_size), torch.split(queries_tensor, batch_size)):  # split into smaller batches to fit GPU memory
            output = model(batch_combined_pc, batch_queries)
            # print(output)
            stress_outputs.append(output[0]) # stress prediction
            occupancy_outputs.append(output[1]) # occupancy prediction

    all_stresses = torch.cat(tuple(stress_outputs), dim=0).squeeze().cpu().detach().numpy()
    all_occupancies = torch.cat(tuple(occupancy_outputs), dim=0).squeeze().cpu().detach().numpy()
    occupied_idxs = np.where(all_occupancies >= occupancy_threshold)[0] # queries that the model thinks belong to the volume of the object
    
    # print(all_stresses.shape, all_occupancies.shape)

    return all_stresses, all_occupancies, occupied_idxs

def sample_points_bounding_box(object_mesh, num_pts, scales=[1.5, 1.5, 1.5], seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Get the bounding box of the mesh
    bbox = object_mesh.bounding_box

    # Get the minimum and maximum coordinates of the bounding box
    min_coords = bbox.bounds[0]
    max_coords = bbox.bounds[1]

    # Calculate the dimensions of the bounding box
    dimensions = max_coords - min_coords

    # Extend the dimensions by a factor of 'scales'
    extended_dimensions = dimensions * np.array(scales)

    # Calculate the center of the extended box
    center = (min_coords + max_coords) / 2.0

    # Calculate the minimum and maximum coordinates of the extended box
    extended_min_coords = center - extended_dimensions / 2.0
    extended_max_coords = center + extended_dimensions / 2.0

    # Sample points within the extended box
    return np.random.uniform(extended_min_coords, extended_max_coords, size=(num_pts, 3))


def sample_and_compute_signed_distance(tri_indices, full_pc, boundary_threshold, num_pts, scales, vis=False, seed=0, verbose=True):
    """ 
    Sample points and decide whether these points are inside, on the surface, or outside the object mesh.
    boundary_threshold: [a,b] where a (positive) is the max distance at which a point is 
                        considered to be on the mesh surface, b (negative) is the min.
                        > a means the query point is inside the object, < b means outside.
    
    """
    
    mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))

    queries = sample_points_bounding_box(mesh, num_pts, scales, seed=seed)
        
    signed_distances = trimesh.proximity.signed_distance(mesh, queries)     # Points OUTSIDE the mesh will have NEGATIVE distance
    outside_mesh_idxs = np.where(signed_distances < boundary_threshold[1])[0]
    
    if verbose:
        print("** num outside:", len(outside_mesh_idxs))
        # print(signed_distances)
    
    if vis:
        mesh = open3d.geometry.TriangleMesh()
        np_vertices = full_pc
        np_triangles = np.array(tri_indices).reshape(-1,3).astype(np.int32)
        mesh.vertices = open3d.utility.Vector3dVector(np_vertices)
        mesh.triangles = open3d.utility.Vector3iVector(np_triangles)
        pcds = [mesh]
        for i, query in enumerate(queries):
            query_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            if i in outside_mesh_idxs:
                color = [1,0,0]
            else:
                color = [0,0,1]
            query_sphere.paint_uniform_color(color)
            query_sphere.translate(tuple(query))    
            pcds.append(query_sphere)
        open3d.visualization.draw_geometries(pcds)     
        
    return queries, outside_mesh_idxs