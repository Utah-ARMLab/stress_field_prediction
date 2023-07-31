import numpy as np
import trimesh
import os
import pickle
import random
import open3d
from copy import deepcopy
import sys
sys.path.append("../")
from utils.hands import create_gripper

def get_mesh_position(idx, num_cols, spacing):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    return [row_idx * spacing, -col_idx * spacing, 0]


gripper = create_gripper('panda', 0.04, franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=False) 
gripper.mesh.visual.face_colors = [250, 0, 0, 255]                            

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset"
# object_names = ["6polygon04", "cuboid01", "cylinder05", "ellipsoid04", "sphere06"]
object_names = [f"cuboid0{j}" for j in [3]]   #[f"6polygon0{j}" for j in [1,4]] + [f"sphere0{j}" for j in [1,2,3,4,5,6]] 
meshes = [gripper.mesh.apply_translation([0.,0,-0.10])]

""" 
Thickness of foams at home: 4.5  cm and 2.5 cm
Panda max width (at full opening): 0.04 + 0.04 = 8 cm 
Selected box dims:
(0.07, 0.05, 0.045)
(0.07, 0.07, 0.06)
(0.07, 0.035, 0.025)
(0.06, 0.045, 0.035)
(0.03, 0.15, 0.03)
(0.07, 0.15, 0.045)
(0.05, 0.20, 0.06)

"""
box_mesh = trimesh.creation.box((0.05, 0.20, 0.06))
meshes.append(box_mesh)

# for i, obj_name in enumerate(object_names):
#     mesh_dir = os.path.join(mesh_main_path, obj_name)
#     fname_object = os.path.join(mesh_dir, f"{obj_name}.stl")
#     mesh = trimesh.load(fname_object)
#     meshes.append(mesh.apply_translation([0.15,0,-0.0]))
    
#     print("mesh.extents:", mesh.extents)

trimesh.Scene(meshes).show()

    






