import numpy as np
# from sklearn.decomposition import PCA
import trimesh

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


def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points



def transform_point_clouds(point_clouds, matrices):

    # If there's only one point cloud but multiple matrices, repeat and reshape point cloud to match matrices shape.
    if len(point_clouds.shape) == 2 and len(matrices.shape) == 3:
        num_matrices = matrices.shape[0]
        point_clouds = np.tile(point_clouds, (num_matrices, 1, 1))
    
    # If there's both only one point cloud and one matrix, add an extra dimension to both.    
    elif len(point_clouds.shape) == 2:
        point_clouds = point_clouds[np.newaxis, ...]
        matrices = matrices[np.newaxis, ...]
    
    # Convert 3D points to homogeneous coordinates
    homogeneous_points = np.concatenate((point_clouds, np.ones_like(point_clouds[..., :1])), axis=-1)
   
    # Perform matrix multiplication for all point clouds at once using broadcasting
    transformed_points = np.matmul(homogeneous_points, matrices.swapaxes(1, 2))
    
    return transformed_points[:,:,:3]