import numpy as np
import trimesh
import os
import sys
sys.path.append("../")
from utils.mesh_utils import simplify_mesh_pymeshlab, create_tet_mesh
import pymeshlab as ml

def get_mesh_position(idx, num_cols, spacing):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    return [row_idx * spacing, -col_idx * spacing, 0]

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"
# mesh_main_path = "/home/baothach/sim_data/BigBird/BigBird_mesh"

# selected_objects = ["windex_bottle", "bleach_cleanser", "mustard_bottle"]
selected_objects = ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol", "mustard_bottle"]


meshes = []
for i, object_name in enumerate(selected_objects):
    # file_name = os.path.join(mesh_main_path, f"{object_name}/google_16k/nontextured_proc.stl")
    file_name = os.path.join(mesh_main_path, f"{object_name}/meshes/poisson_proc.stl")
    
    mesh = trimesh.load(file_name)
    # mesh.show()

    mesh = simplify_mesh_pymeshlab(mesh, target_num_vertices=250)       
    print(mesh.vertices.shape, mesh.faces.shape)
    
    # save_mesh_fname = os.path.join(mesh_main_path, f"{object_name}/{object_name}.stl")
    # mesh.export(save_mesh_fname)
    # create_tet_mesh(os.path.join(mesh_main_path, f"{object_name}"), object_name, coarsen=True, verbose=True)
    
    meshes.append(mesh.apply_translation(get_mesh_position(i, num_cols=2, spacing=0.1)))
    
    # break
    
trimesh.Scene(meshes).show()