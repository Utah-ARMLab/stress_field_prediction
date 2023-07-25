import sys
from utils.graspsampling_utilities import poses_wxyz_to_mats
from utils.hands import create_gripper
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb
import trimesh
import torch
import open3d
import numpy as np
from scipy.stats import halfnorm

def get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=1024, gripper_name='panda', finger_only=True):
    gripper = create_gripper(gripper_name, configuration=fingers_joint_angles, 
                             franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=finger_only)
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

def get_stress_occupancy_query_points(model, device, query_list, combined_pc, batch_size, occupancy_threshold):
    """ 
    query_list: np array, list of query points. Shape (num_qrs, 3). 
    combined_pc: augmented partial pc concatenated with gripper pc. Shape (1024+1024, 5).
 
    Returns: (O1, O2, O3)
    O1: np array, list of predicted stress scalars at each query points. Shape (num_qrs,).   
    O2: np array, confidence score [0,1] of whether each query point belongs to the volume of the object. Shape (num_qrs,). 
    O3: indices of all query points that the model is confident to belong to the volume of the object (likelihood >= occupancy_threshold). 
    """
    
    num_qrs = query_list.shape[0]
    queries_tensor = torch.from_numpy(query_list).float().to(device)  # shape (num_qrs, 3)
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

def sample_points_bounding_box(object, num_pts, scales=[1.5, 1.5, 1.5], seed=None):
    """ 
    Sample points from the bounding box of the object mesh.
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(object, trimesh.Trimesh):
        # Get the bounding box of the mesh
        bbox = object.bounding_box

        # Get the minimum and maximum coordinates of the bounding box
        min_coords = bbox.bounds[0]
        max_coords = bbox.bounds[1]
    
    elif isinstance(object, trimesh.PointCloud):
        min_coords = object.bounds[0]
        max_coords = object.bounds[1]        

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


def sample_points_gaussian(object_mesh, num_pts, scales=[1.5, 1.5, 1.5], tolerance=0.001, seed=None):
    """ 
    Sample points from object mesh surface. Then add to each point some half-gaussian noise (only positive).
    tolerance: minimum signed distance of a query point to be considered to belong to the object volume.
    """

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

    # Sample epsilon and shift the surface points in the direction of the normal vectors.
    sigma = max(extended_dimensions - dimensions) / 2
    surface_points, faces = trimesh.sample.sample_surface_even(object_mesh, count=round(num_pts*1.5))
    normals = object_mesh.face_normals[faces[:num_pts]]
    dist = halfnorm(scale=sigma)
    epsilon = dist.rvs(size=num_pts)  # distance epsilon to shift the surface points. Epsilon is sampled of a half-gaussian distribution. https://stats.stackexchange.com/questions/603768/how-do-you-sample-from-a-half-normal-distribution-in-python
    noisy_points = surface_points[:num_pts] + epsilon[:, np.newaxis] * normals

    # is_inside = None
    # # noisy_points = surface_points[:num_pts] + (0.001 * np.ones((num_pts,1))) * normals  # constant 1mm epsilon example
    signed_distances_noisy_points = trimesh.proximity.signed_distance(object_mesh, noisy_points)
    is_inside = signed_distances_noisy_points >= -tolerance
    # print(np.sum(is_inside))
    # # print(min(signed_distances_noisy_points), max(signed_distances_noisy_points))
    

    return noisy_points, is_inside#, signed_distances_noisy_points

def sample_points_gaussian_2(object_mesh, num_pts, scales=[1.5, 1.5, 1.5], tolerance=0.001, seed=None):
    """ 
    Sample points from object mesh surface. Then add to each point some half-gaussian noise (only positive).
    Also add sub-surface points: points on the surface added with the small tolerance, labelled as un-occupied.
    tolerance: minimum signed distance of a query point to be considered to belong to the object volume.
    """

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

    num_pts = num_pts // 2

    # Sample epsilon and shift the surface points in the direction of the normal vectors.
    sigma = max(extended_dimensions - dimensions) / 2
    surface_points, faces = trimesh.sample.sample_surface_even(object_mesh, count=round(num_pts*1.5))
    normals = object_mesh.face_normals[faces[:num_pts]]
    dist = halfnorm(scale=sigma)
    epsilon = dist.rvs(size=num_pts)  # distance epsilon to shift the surface points. Epsilon is sampled of a half-gaussian distribution. https://stats.stackexchange.com/questions/603768/how-do-you-sample-from-a-half-normal-distribution-in-python
    noisy_points = surface_points[:num_pts] + epsilon[:, np.newaxis] * normals

    # is_inside = None
    # # noisy_points = surface_points[:num_pts] + (0.001 * np.ones((num_pts,1))) * normals  # constant 1mm epsilon example
    signed_distances_noisy_points = trimesh.proximity.signed_distance(object_mesh, noisy_points)
    is_inside = signed_distances_noisy_points >= -tolerance
    # print(np.sum(is_inside))
    # # print(min(signed_distances_noisy_points), max(signed_distances_noisy_points))
    
    sub_surface_points = surface_points[:num_pts] + tolerance * normals
    noisy_points = np.concatenate((noisy_points, sub_surface_points), axis=0)
    is_inside = np.concatenate((is_inside, np.full((num_pts,), False, dtype=bool)), axis=0)

    return noisy_points, is_inside

def sample_points_gaussian_3(object_mesh, num_pts, scales=[1.5, 1.5, 1.5], tolerance=0.001, seed=None):
    """ 
    Sample points from object mesh surface. Then add to each point some half-gaussian noise (only positive).
    Also, ensure that all query points are un-occupied, by ignoring the occupied points.
    object_mesh: surface mesh.
    tolerance: minimum signed distance of a query point to be considered to belong to the object volume.
    """

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

    # Sample epsilon and shift the surface points in the direction of the normal vectors.
    sigma = max(extended_dimensions - dimensions) / 2
    surface_points, faces = trimesh.sample.sample_surface_even(object_mesh, count=round(num_pts*1.5))
    normals = object_mesh.face_normals[faces]
    dist = halfnorm(scale=sigma)
    epsilon = dist.rvs(size=surface_points.shape[0])  # distance epsilon to shift the surface points. Epsilon is sampled of a half-gaussian distribution. https://stats.stackexchange.com/questions/603768/how-do-you-sample-from-a-half-normal-distribution-in-python
    noisy_points = surface_points + epsilon[:, np.newaxis] * normals

    signed_distances_noisy_points = trimesh.proximity.signed_distance(object_mesh, noisy_points)
    is_inside = signed_distances_noisy_points >= -tolerance
    
    return noisy_points[np.where(is_inside==False)[0]][:num_pts]

    

    return noisy_points, is_inside#, signed_distances_noisy_points

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
        # pcds = [mesh]
        # for i, query in enumerate(queries):
        #     query_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #     if i in outside_mesh_idxs:
        #         color = [1,0,0]
        #     else:
        #         color = [0,0,1]
        #     query_sphere.paint_uniform_color(color)
        #     query_sphere.translate(tuple(query))    
        #     pcds.append(query_sphere)
        # open3d.visualization.draw_geometries(pcds)     

        pcd_outside = pcd_ize(queries[outside_mesh_idxs], color=[1,0,0])   
        inside_mesh_idxs = np.where(signed_distances >= boundary_threshold[1])[0]           
        pcd_inside = pcd_ize(queries[inside_mesh_idxs], color=[0,1,0])
        open3d.visualization.draw_geometries([pcd_outside.translate((0.09,0,0)), pcd_inside.translate((-0.09,0,0)), mesh])
        
    return queries, signed_distances, outside_mesh_idxs


# def reconstruct_stress_field(model, device, batch_size, tri_indices, occupancy_threshold, full_pc, 
#                              combined_pc, query_type, num_query_pts=None, 
#                              stress_visualization_bounds=None, return_open3d_object=False):
#     if query_type == "sampled": # sampled from bounding box
#         sampled_points, outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
#                                         boundary_threshold=[0.02,-0.01], \
#                                         num_pts=num_query_pts, scales=[1.5, 1.5, 1.5], vis=False, seed=None, verbose=False)  
#         predicted_stresses_log, predicted_occupancies, occupied_idxs = \
#                                 get_stress_occupancy_query_points(model, device, query_list=sampled_points, 
#                                                                   combined_pc=combined_pc,
#                                                                   batch_size=batch_size, occupancy_threshold=occupancy_threshold)
#         print("occupied_idxs.shape:", occupied_idxs.shape)
#         selected_points = sampled_points[occupied_idxs] # select points that belong to the volume of the object (occupied)
#         selected_stresses_log = predicted_stresses_log[occupied_idxs]

#     elif query_type == "full":   # use object particles ("full_pc")
#         predicted_stresses_log, predicted_occupancies, occupied_idxs = \
#                                 get_stress_occupancy_query_points(model, device, query_list=full_pc, 
#                                                                   combined_pc=combined_pc,
#                                                                   batch_size=batch_size, occupancy_threshold=occupancy_threshold)
#         selected_points = full_pc
#         selected_stresses_log = predicted_stresses_log

#     if not return_open3d_object:
#         return selected_points
#     else:
#         pcd_stress_field = pcd_ize(selected_points)
#         if stress_visualization_bounds is None:
#             stress_field_colors = np.array(scalar_to_rgb(selected_stresses_log, colormap='jet'))[:,:3]
#         else:
#             stress_field_colors = np.array(scalar_to_rgb(
#                 selected_stresses_log, colormap='jet', 
#                 min_val=stress_visualization_bounds[0], max_val=stress_visualization_bounds[1]))[:,:3]
            
#         pcd_stress_field.colors = open3d.utility.Vector3dVector(stress_field_colors)        

#         return pcd_stress_field

def Tetrahedron_vectorized(vertices):
    """
    Given a list of the xyz coordinates of the vertices of a tetrahedron, 
    return tetrahedron coordinate system
    """
    origin = vertices[:,:1,:]
    rest = vertices[:,1:,:]
    mat = (np.array(rest) - origin)
    tetra = np.linalg.inv(mat)
    return tetra.transpose((0,2,1)), origin

def pointInside_vectorized(point, tetra, origin):
    """
    Takes a single point or array of points, as well as tetra and origin objects returned by 
    the Tetrahedron function.
    Returns a boolean or boolean array indicating whether the point is inside the tetrahedron.
    """
    newp = np.matmul(tetra, (point-origin).transpose((0,2,1))).transpose((0,2,1))
    return np.all(newp>=0, axis=-1) & np.all(newp <=1, axis=-1) & (np.sum(newp, axis=-1) <=1)


def is_inside_tet_mesh_vectorized(points, vertices, tet_indices):
    """ 
    Whether a set of points belong to the tetrahedral mesh volume.
    Original implementation: https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/60745339#60745339      
    """
    v = vertices[tet_indices]
    tetra, origin = Tetrahedron_vectorized(v)            
    insides = pointInside_vectorized(points, tetra, origin).squeeze()   # shape (num_tetra, points.shape[0])

    insides = np.any(insides, axis=0)   # shape (points.shape[0],)
    
    return insides


def reconstruct_stress_field(model, device, batch_size, tri_indices, occupancy_threshold, full_pc, 
                             combined_pc, query_type, num_query_pts=None, 
                             stress_visualization_bounds=None, return_open3d_object=False):
    if query_type == "sampled": # sampled from bounding box
        sampled_points, outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                        boundary_threshold=[0.02,-0.01], \
                                        num_pts=num_query_pts, scales=[1.5, 1.5, 1.5], vis=False, seed=None, verbose=False)  
        predicted_stresses_log, predicted_occupancies, occupied_idxs = \
                                get_stress_occupancy_query_points(model, device, query_list=sampled_points, 
                                                                  combined_pc=combined_pc,
                                                                  batch_size=batch_size, occupancy_threshold=occupancy_threshold)
        print("occupied_idxs.shape:", occupied_idxs.shape)
        selected_points = sampled_points[occupied_idxs] # select points that belong to the volume of the object (occupied)
        selected_stresses_log = predicted_stresses_log[occupied_idxs]

    elif query_type == "full":   # use object particles ("full_pc")
        predicted_stresses_log, predicted_occupancies, occupied_idxs = \
                                get_stress_occupancy_query_points(model, device, query_list=full_pc, 
                                                                  combined_pc=combined_pc,
                                                                  batch_size=batch_size, occupancy_threshold=occupancy_threshold)
        selected_points = full_pc
        selected_stresses_log = predicted_stresses_log

    if not return_open3d_object:
        return selected_points
    else:
        pcd_stress_field = pcd_ize(selected_points)
        if stress_visualization_bounds is None:
            stress_field_colors = np.array(scalar_to_rgb(selected_stresses_log, colormap='jet'))[:,:3]
        else:
            stress_field_colors = np.array(scalar_to_rgb(
                selected_stresses_log, colormap='jet', 
                min_val=stress_visualization_bounds[0], max_val=stress_visualization_bounds[1]))[:,:3]
            
        pcd_stress_field.colors = open3d.utility.Vector3dVector(stress_field_colors)        

        return pcd_stress_field