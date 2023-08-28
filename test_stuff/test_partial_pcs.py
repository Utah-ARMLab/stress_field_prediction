import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import random
import torch
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb, read_pickle_data, print_color
from utils.point_cloud_utils import transform_point_cloud, is_homogeneous_matrix, world_to_object_frame
from copy import deepcopy
import re
import trimesh

# gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/all_primitives"
gripper_pc_recording_path = os.path.join(data_main_path,  f"processed")



selected_objects = []
# selected_objects += \
# [f"lemon0{j}" for j in [1,2,3]] + \
# [f"strawberry0{j}" for j in [1,2,3]] + \
# [f"tomato{j}" for j in [1]] + \
# [f"apple{j}" for j in [3]] + \
# [f"potato{j}" for j in [3]]
# selected_objects += ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol"]
# selected_objects += [f"cylinder0{j}" for j in range(1,9)] + [f"box0{j}" for j in range(1,10)] \
#                 + [f"ellipsoid0{j}" for j in range(1,6)] + [f"sphere0{j}" for j in [1,3,4,6]]

# "mustard_bottle"

selected_objects += [f"lemon02", "hemi01"]

pcds = []
trimesh_pcs = []
for idx, file_name in enumerate(selected_objects):
    object_name = os.path.splitext(file_name)[0]
    prim_name = re.search(r'(\D+)', object_name).group(1)
    data_recording_path = os.path.join(data_main_path, f"all_{prim_name}_data")

    ### Load static data
    static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                    f"{object_name}.pickle"))   # shape (8, num_pts, 3)
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    homo_mats = static_data["homo_mats"]
    tet_indices = static_data["tet_indices"]
    
    partial_pcs = static_data["partial_pcs"]  
    pc = partial_pcs[3]
    pc -= np.mean(pc, axis=0)
    
    # partial_pcs = static_data["transformed_partial_pcs"]  
    # pc = partial_pcs[0] 
    
    # pcds.append(pcd_ize(pc))

    homo_mat = world_to_object_frame(pc)
    trimesh_pc = trimesh.PointCloud(pc)
    trimesh_pc.apply_transform(homo_mat)
    trimesh_pcs.append(trimesh_pc)
    # trimesh_pc.show()
    
    
    
# pcds[0].paint_uniform_color([0,0,0])
# pcds[1].paint_uniform_color([1,0,0])
# open3d.visualization.draw_geometries(pcds)

trimesh_pcs[0].apply_translation([0.05,0,0])
trimesh.Scene(trimesh_pcs).show()