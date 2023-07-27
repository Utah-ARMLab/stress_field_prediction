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


from sklearn.decomposition import PCA
import numpy as np

import pdb

def find_pca_axes(obj_cloud, verbose=False):
    '''
    Given a point cloud determine a valid, right-handed coordinate frame
    '''
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(obj_cloud)
    centroid = np.matrix(pca_operator.mean_).T
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)

    if verbose:
        print('PCA centroid', centroid)
        print('x_axis', x_axis)
        print('y_axis', y_axis)
        print('z_axis', z_axis)
    return np.array([x_axis, y_axis, z_axis]), centroid

#Compute angles between two vectors, code is from:
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
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


def homogeneous_transform_to_object_frame(obj_cloud, verbose=False):
    '''
    For the given object cloud, build an object frame using PCA and aligning to the
    world frame.
    Returns a transformation from world frame to object frame.
    '''

    # Use PCA to find a starting object frame/centroid.
    axes, centroid = find_pca_axes(obj_cloud, verbose)
    axes = np.matrix(np.column_stack(axes))

    # Rotation from object frame to frame.
    R_o_w = np.eye(3)
    
    #Find and align x axes.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)
    R_o_w[0, 0] = align_x_axis[0, 0]
    R_o_w[1, 0] = align_x_axis[1, 0]
    R_o_w[2, 0] = align_x_axis[2, 0]

    #y axes
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)
    R_o_w[0, 1] = align_y_axis[0, 0]
    R_o_w[1, 1] = align_y_axis[1, 0]
    R_o_w[2, 1] = align_y_axis[2, 0]

    #z axes
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    R_o_w[0, 2] = align_z_axis[0, 0]
    R_o_w[1, 2] = align_z_axis[1, 0]
    R_o_w[2, 2] = align_z_axis[2, 0]

    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, centroid)
    
    # Build full transformation matrix.
    align_trans_matrix = np.eye(4)
    align_trans_matrix[:3,:3] = R_w_o
    align_trans_matrix[0,3] = d_w_o_o[0]
    align_trans_matrix[1,3] = d_w_o_o[1]
    align_trans_matrix[2,3] = d_w_o_o[2]    

    
    assert is_homogeneous_matrix(align_trans_matrix)

    return align_trans_matrix


def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points



static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"  # _original

### Get static data
object_name = "6polygon06"  #6polygon04 ellipsoid01
with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
    static_data = pickle.load(handle)
partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)

coor_objects = []

for i in range(8):
    print(f"View {i}")
    pc = partial_pcs[i]
    coor_global = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

    homo_mat = homogeneous_transform_to_object_frame(pc)
#     coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
#     coor_object.transform(homo_mat)
    pc_transformed = transform_point_cloud(pc, homo_mat)
#     # print(homo_mat)


    pcd = pcd_ize(pc, color=[0,0,0])
    pcd_transformed = pcd_ize(pc_transformed, color=[1,0,0])
#     # open3d.visualization.draw_geometries([pcd, pcd_transformed, coor_global, coor_object])
    open3d.visualization.draw_geometries([pcd, pcd_transformed, coor_global])
    
#     coor_objects.append(coor_object)
    
# open3d.visualization.draw_geometries(coor_objects) 
    

# test_pc = np.random.uniform(size=(4000,3))
# transformed_pcs_1 = []

# start_time = timeit.default_timer()
# for _ in range(8):
#     transform_point_cloud(test_pc, homo_mat)
#     # transformed_pcs_1.append(transform_point_cloud(test_pc, homo_mat))
# # transformed_pcs_1 = np.concatenate(tuple(transformed_pcs_1), axis=0)
# # print("transformed_pcs_1.shape:", transformed_pcs_1.shape)

# print("Method 1 computation time: ", timeit.default_timer() - start_time)

# test_pcs = np.random.uniform(size=(8,4000,3))
# homo_mats = [homo_mat] * 8
# start_time = timeit.default_timer()
# transform_point_clouds_vectorized(test_pcs, homo_mats)
# print("Method 2 computation time: ", timeit.default_timer() - start_time)