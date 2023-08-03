import numpy as np
import trimesh
import os
import pickle
import random
import open3d
from copy import deepcopy
import sys
sys.path.append("../")
from utils.mesh_utils import create_tet_mesh, simplify_mesh
import pymeshlab as ml



box_dataset_geometries = \
[(0.075, 0.06, 0.045),
(0.07, 0.07, 0.06),
(0.07, 0.035, 0.025),
(0.06, 0.045, 0.035),
(0.03, 0.15, 0.03),
(0.07, 0.15, 0.045),
(0.05, 0.20, 0.07)]

# [(0.075, 0.06, 0.045),
# (0.07, 0.07, 0.06),
# (0.07, 0.035, 0.025),
# (0.06, 0.045, 0.035),
# (0.03, 0.15, 0.03),
# (0.07, 0.15, 0.045),
# (0.05, 0.20, 0.07)]

visualization = False
export_mesh = True
mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"

for i, geometry in enumerate(box_dataset_geometries):

    obj_name = f"box0{i+1}"
    print(f"object name: {obj_name}")
    
    mesh = trimesh.creation.box(geometry)
    print(mesh.vertices.shape, mesh.faces.shape)
    
    # pymeshlab_mesh = ml.Mesh(mesh.vertices, mesh.faces)
    # mesh = trimesh.Trimesh(vertices=pymeshlab_mesh.vertex_matrix(), faces=pymeshlab_mesh.face_matrix())
    # print(mesh.vertices.shape, mesh.faces.shape)
    
    mesh_dir = os.path.join(mesh_main_path, obj_name)
    os.makedirs(mesh_dir, exist_ok=True)    
    save_mesh_fname = os.path.join(mesh_dir, f"{obj_name}.stl")    
    
    if export_mesh:
        mesh.export(save_mesh_fname)        
        create_tet_mesh(mesh_dir, obj_name, coarsen=True, verbose=True)
    
    if visualization:
        mesh.show()
        
    # if i >= 5:
    #     break



# for obj_name in OBJECT_NAMES[4:]:
#     mesh_dir = os.path.join(mesh_main_path, obj_name)
#     os.makedirs(mesh_dir, exist_ok=True)
       
#     if visualization:
#         fname_object = os.path.join(mesh_dir, f"{obj_name}.obj")
#         mesh = trimesh.load(fname_object)
#         mesh.show()
        
#     create_tet_mesh(mesh_dir, obj_name)
    






