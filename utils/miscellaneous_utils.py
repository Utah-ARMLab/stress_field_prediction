import open3d
import numpy as np
from copy import deepcopy
from utils.farthest_point_sampling import *
import trimesh
import transformations
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch
from isaacgym import gymapi
import pickle
from isaacgym import gymtorch
from utils.camera_utils import get_partial_pointcloud_vectorized


def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd




def get_object_particle_state(gym, sim, vis=False):
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