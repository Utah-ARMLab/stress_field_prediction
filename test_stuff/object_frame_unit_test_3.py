import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, read_pickle_data
import os
import pickle
import open3d
from copy import deepcopy
import timeit
import trimesh

def apply_random_transformation(point_cloud, translation_range=(-0.05, 0.05), rotation_range=(-np.pi, np.pi)):
    """
    Apply a random transformation to a 3D point cloud.

    Parameters:
        point_cloud (numpy.ndarray): A 2D array of shape (N, 3) representing the 3D point cloud.
        translation_range (tuple): Range of translation along each axis (default: (-1, 1)).
        rotation_range (tuple): Range of rotation angles in radians around each axis (default: (-pi, pi)).

    Returns:
        transformed_point_cloud (numpy.ndarray): A 2D array of shape (N, 3) representing the transformed point cloud.
    """
    # Apply translation
    translation = np.random.uniform(translation_range[0], translation_range[1], size=(3,))
    point_cloud += translation

    # Apply rotation
    rotation_angles = np.random.uniform(rotation_range[0], rotation_range[1], size=(3,))
    rotation_matrix = get_rotation_matrix(rotation_angles)
    point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return point_cloud

def get_rotation_matrix(angles):
    """
    Get a 3D rotation matrix given rotation angles around each axis.

    Parameters:
        angles (numpy.ndarray): A 1D array of shape (3,) representing rotation angles in radians around each axis.

    Returns:
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
    """
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.dot(np.dot(Rz, Ry), Rx)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx

def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.e-6) or \
            not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.e-6):
        
        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)        
        print(np.linalg.det(rotational_matrix))
        
        return False

    return True

def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points



def object_to_world_frame(points):

    """  
    Compute 4x4 homogeneous transformation matrix to transform object frame to world frame. 
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3,:3]   # x, y, z axes concat together

    # Find and align z axis
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align x axis.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align y axis
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 


    homo_mat[:3,:3] = np.column_stack((align_x_axis, align_y_axis, align_z_axis))

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


def world_to_object_frame(points):

    """  
    Compute 4x4 homogeneous transformation matrix to transform world frame to object frame. 
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.
    
    **This function is used to define a new frame for the object point cloud. Crucially, it creates the training data and defines the pc for test time.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3,:3]   # x, y, z axes concat together

    # Find and align z axis
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align x axis.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align y axis
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 

    R_o_w = np.column_stack((align_x_axis, align_y_axis, align_z_axis))
    
    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, homo_mat[:3,3])
    
    homo_mat[:3,:3] = R_w_o
    homo_mat[:3,3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"  # _original

### Get static data
object_name = "hemi01"  #6polygon04 ellipsoid01
with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
    static_data = pickle.load(handle)
partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)
pcd_partial = pcd_ize(np.concatenate(tuple(partial_pcs), axis=0), color=[0,0,0])

coor_objects = []

for i in range(1):
    print(f"View {i}")
    pc = partial_pcs[i] #+ np.array([0.2,0,0])
    # pc = apply_random_transformation(pc)

    coor_global = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)

    pcd = pcd_ize(pc, color=[0,0,0])
    homo_mat = object_to_world_frame(pc)

    coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    coor_object.transform(homo_mat)    
    # open3d.visualization.draw_geometries([pcd, coor_global, coor_object])

    homo_mat = world_to_object_frame(pc)
    pc_transformed = transform_point_cloud(pc, homo_mat)
    pcd_transformed = pcd_ize(pc_transformed, color=[1,0,0])
    open3d.visualization.draw_geometries([pcd, pcd_transformed, coor_global, coor_object])
    # open3d.visualization.draw_geometries([coor_global, coor_object, pcd])
    # # # open3d.visualization.draw_geometries([pcd, coor_object])
    
    coor_objects.append(coor_object)
coor_objects.append(pcd_partial)    
# open3d.visualization.draw_geometries(coor_objects) 
    

