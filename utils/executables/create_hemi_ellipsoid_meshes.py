import pickle
import open3d
import numpy as np
import os
import trimesh
from copy import deepcopy
import sys
sys.path.append("../../")
from utils.mesh_utils import trimesh_to_open3d_mesh, create_tet_mesh
from utils.point_cloud_utils import world_to_object_frame, transform_point_cloud
from utils.miscellaneous_utils import get_object_particle_state, pcd_ize

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"
vis = False

radii = [0.028, 0.04]

for i, radius in enumerate(radii):

    object_name = f"hemi0{i+1}"
    os.makedirs(os.path.join(mesh_main_path, f"{object_name}"), exist_ok=True)
    print(f"object name: {object_name}")

    hemis_mesh = trimesh.creation.icosphere(radius = radius)     # hemi-ellipsoid that matches (almost) exactly with the shape of lemon02  
    hemis_mesh = trimesh.intersections.slice_mesh_plane(mesh=hemis_mesh, plane_normal=[0,-1,0], plane_origin=[0,0.00,0], cap=True)
    hemis_mesh.visual.face_colors = [250, 0, 0, 255]
    hemis_mesh.vertices *= np.array([5./6,1.1,5./6])
    
    hemis_mesh.vertices -= hemis_mesh.center_mass

    save_mesh_fname = os.path.join(mesh_main_path, f"{object_name}/{object_name}.stl")
    hemis_mesh.export(save_mesh_fname)    
    create_tet_mesh(os.path.join(mesh_main_path, f"{object_name}"), object_name, coarsen=True, verbose=True)
    
    if vis:
        lemon_mesh_path = os.path.join(mesh_main_path, "lemon02", f"lemon02_processed.stl")
        lemon_mesh = trimesh.load(lemon_mesh_path)    
        lemon_mesh.vertices -= lemon_mesh.center_mass   #np.mean(mesh.vertices, axis=0)
        lemon_mesh.visual.face_colors = [0, 0, 250, 128]

        lemon_mesh.apply_translation([0.05,0,0])

        # # Define the rotation angle in radians
        # angle = np.radians(30)  # np.radians(30) 

        # # Create a rotation matrix around the X-axis
        # rotation_matrix = np.array([
        #     [1, 0, 0, 0],
        #     [0, np.cos(angle), -np.sin(angle), 0],
        #     [0, np.sin(angle), np.cos(angle), 0],
        #     [0, 0, 0, 1]
        # ])

        # # Apply the rotation to the mesh
        # hemis_mesh.apply_transform(rotation_matrix)
        hemis_mesh.vertices -= hemis_mesh.center_mass   # np.mean(hemis_mesh.vertices, axis=0)


        trimesh.Scene([lemon_mesh, hemis_mesh]).show()