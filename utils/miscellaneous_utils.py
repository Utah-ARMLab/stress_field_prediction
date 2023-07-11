import open3d
import numpy as np
from copy import deepcopy
# from utils.farthest_point_sampling import *
import trimesh
import transformations
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle



def down_sampling(pc, num_pts=1024, return_indices=False):
    # farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    # pc = pc[farthest_indices.squeeze()]  
    # return pc

    """
    Input:
        pc: pointcloud data, [B, N, D] where B= num batches, N=num points, D=feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled pointcloud index, [num_pts, D]
    """

    if pc.ndim == 2:
        # insert batch_size axis
        pc = deepcopy(pc)[None, ...]

    B, N, D = pc.shape
    xyz = pc[:, :,:3]
    centroids = np.zeros((B, num_pts))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)

    pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]

    if return_indices:
        return pc.squeeze(), centroids.astype(np.int32)

    return pc.squeeze()


def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd




def get_object_particle_state(gym, sim, vis=False):
    from isaacgym import gymtorch
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    particles = particle_state_tensor.numpy()[:, :3]  
    
    if vis:
        pcd_ize(particles,vis=True)
    
    return particles.astype('float32')


# def record_data_stress_prediction(gym, sim, env, \
#                                 undeformed_object_pc, undeformed_object_particle_state, undeformed_gripper_pc, current_desired_force, \
#                                 object_name, object_young_modulus, object_scale, \
#                                 cam_handle, cam_prop, robot_segmentationId=11, min_z=0.01, vis=False):
#     ### Get current object pc and particle position:
#     object_pc = get_partial_pointcloud_vectorized(gym, sim, env, cam_handle, cam_prop, robot_segmentationId, "deformable", None, min_z, vis, device="cpu")
#     object_particle_position = get_object_particle_state(gym, sim)

def record_data_stress_prediction(data_recording_path, gym, sim, 
                                current_desired_force, grasp_pose, fingers_joint_angles, 
                                object_name, young_modulus, object_scale):
                                    
    ### Get current object particle state:
    object_particle_state = get_object_particle_state(gym, sim)
       
    data = {"object_particle_state": object_particle_state, "force": current_desired_force, 
        "grasp_pose": grasp_pose, "fingers_joint_angles": fingers_joint_angles, 
        "tet": gym.get_sim_tetrahedra(sim), "tri": gym.get_sim_triangles(sim), 
        "object_name": object_name, "young_modulus": young_modulus, "object_scale": object_scale}    
    
    with open(data_recording_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def normalize_list(lst):
    minimum = min(lst)
    maximum = max(lst)
    value_range = maximum - minimum

    normalized_lst = [(value - minimum) / value_range for value in lst]

    return normalized_lst


def scalar_to_rgb(scalar_list, colormap='jet', min_val=None, max_val=None):
    if min_val is None:
        norm = plt.Normalize(vmin=np.min(scalar_list), vmax=np.max(scalar_list))
    else:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.cm.get_cmap(colormap)
    rgb = cmap(norm(scalar_list))
    return rgb




def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)