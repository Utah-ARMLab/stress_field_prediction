import numpy as np
import trimesh
import os
import pickle
import random
import open3d
from copy import deepcopy
from constants import OBJECT_NAMES
import sys
sys.path.append("../")
from utils.hands import create_gripper

def get_mesh_position(idx, num_cols, spacing):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    return [row_idx * spacing, -col_idx * spacing, 0]


gripper = create_gripper('panda', 0.00, franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=True) 
                            

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"
# object_names = ["6polygon04", "cuboid01", "cylinder05", "ellipsoid04", "sphere06"]
object_names = [f"box0{j}" for j in [1,2,3,4,5,6,7]]   #[f"6polygon0{j}" for j in [1,4]] + [f"sphere0{j}" for j in [1,2,3,4,5,6]] 
# meshes = [gripper.mesh.apply_translation([-0.,0,0])]
meshes = []

for i, obj_name in enumerate(object_names):
    mesh_dir = os.path.join(mesh_main_path, obj_name)
    fname_object = os.path.join(mesh_dir, f"{obj_name}.stl")
    mesh = trimesh.load(fname_object)
    meshes.append(mesh.apply_translation(get_mesh_position(i, num_cols=2, spacing=0.1)))
    
    # print("mesh.extents:", mesh.extents)

trimesh.Scene(meshes).show()

    






