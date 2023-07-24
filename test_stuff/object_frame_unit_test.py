import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, read_pickle_data
import os
import pickle
import open3d
from copy import deepcopy

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

# def homogeneous_transform_to_object_frame(points):
#     # Step 1: Calculate the centroid of the points
#     centroid = np.mean(points, axis=0)

#     # Step 2: Use PCA to find the dominant axis (x-axis)
#     pca = PCA(n_components=1)
#     pca.fit(points)
#     x_axis = pca.components_.flatten()
#     x_axis = np.round(x_axis, decimals=2)
#     x_axis /= np.linalg.norm(x_axis)
    

#     # Step 3: Calculate the y-axis using the cross product of x and z axes
#     z_axis = np.array([0., 0., 1.])
#     # y_axis = np.cross(z_axis, x_axis) 
#     # y_axis /= np.linalg.norm(y_axis)
#     y_axis = np.array([z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
#                        -(z_axis[0]*x_axis[2] - z_axis[2]*x_axis[0]),
#                        z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]])
#     print(np.linalg.norm(y_axis))

#     # Step 4: Create the transformation matrix
#     rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
#     translation_vector = centroid

#     # Combine rotation and translation into a 4x4 homogeneous transformation matrix
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = rotation_matrix
#     transformation_matrix[:3, 3] = translation_vector

#     return transformation_matrix

def homogeneous_transform_to_object_frame(points):
    
    # Step 1: Use PCA to find the dominant axis (x-axis, y-axis)
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(points)
    centroid = pca_operator.mean_
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)
    # print(z_axis)
    
    if z_axis[2] < 0:
        if x_axis[0] < 0:
            x_axis *= -1
        else:
            y_axis *= -1
        z_axis *= -1
        

    # Step 4: Create the transformation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    translation_vector = centroid

    # Combine rotation and translation into a 4x4 homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    
    assert is_homogeneous_matrix(transformation_matrix)

    return transformation_matrix


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
object_name = "6polygon04"  #6polygon04 ellipsoid01
with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
    static_data = pickle.load(handle)
partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)

coor_objects = []

for i in range(8):
    print(f"View {i}")
    pc = partial_pcs[i]
    coor_global = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

    homo_mat = homogeneous_transform_to_object_frame(pc)
    coor_object = deepcopy(coor_global).transform(homo_mat)
    pc_transformed = transform_point_cloud(pc, homo_mat)
    # print(homo_mat)


    pcd = pcd_ize(pc, color=[0,0,0])
    pcd_transformed = pcd_ize(pc_transformed, color=[1,0,0])
    # open3d.visualization.draw_geometries([pcd, pcd_transformed, coor_global, coor_object])
    
    coor_objects.append(coor_object)
    
open3d.visualization.draw_geometries(coor_objects) 
    
    

