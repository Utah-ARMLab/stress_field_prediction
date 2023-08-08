import pickle
import open3d
import numpy as np
import os
import trimesh
import sys
sys.path.append("../")
from utils.mesh_utils import trimesh_to_open3d_mesh

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"

object_names = \
[f"tomato{j}" for j in [1,2]] + \
[f"potato{j}" for j in [1,2,3]] + \
[f"eggplant{j}" for j in [1,2]] + \
[f"apple{j}" for j in [1,2,3]] + \
[f"cucumber{j}" for j in [1]] + \
[f"strawberry0{j}" for j in [1,2,3]] + \
[f"lemon0{j}" for j in [1,2,3]] 

selected_objects = \
[f"lemon0{j}" for j in [1,2,3]] + \
[f"strawberry0{j}" for j in [1,2,3]] + \
[f"tomato{j}" for j in [1]] + \
[f"apple{j}" for j in [3]] + \
[f"potato{j}" for j in [3]]

# selected_fruit_names = ["potato"]
# selected_fruit_names = ["apple", "lemon", "potato", "strawberry", "tomato"]
# selected_fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]

# for object_name in object_names:

    # if not any([fruit_name in object_name for fruit_name in selected_fruit_names]):   # if object does NOT belong to any of the selected primitives.
    #     continue

for object_name in selected_objects:


    print(f"object name: {object_name}")
    

    mesh_path = os.path.join(mesh_main_path, object_name, f"{object_name}_processed.stl")
    mesh = trimesh.load(mesh_path)
    print("Extents of loaded mesh:", mesh.extents)
    object_mesh = trimesh_to_open3d_mesh(mesh)
    

    coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    open3d.visualization.draw_geometries([object_mesh, coor]) 

    print("========================")

    # break