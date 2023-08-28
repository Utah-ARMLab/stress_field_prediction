import pickle
import open3d
import numpy as np
import os
import trimesh
from copy import deepcopy
import sys
sys.path.append("../")
from utils.mesh_utils import trimesh_to_open3d_mesh
from utils.point_cloud_utils import world_to_object_frame, transform_point_cloud
from utils.miscellaneous_utils import get_object_particle_state, pcd_ize

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"

object_names = \
[f"tomato{j}" for j in [1,2]] + \
[f"potato{j}" for j in [1,2,3]] + \
[f"eggplant{j}" for j in [1,2]] + \
[f"apple{j}" for j in [1,2,3]] + \
[f"cucumber{j}" for j in [1]] + \
[f"strawberry0{j}" for j in [1,2,3]] + \
[f"lemon0{j}" for j in [1,2,3]] 

# selected_objects = \
# [f"lemon0{j}" for j in [1,2,3]] + \
# [f"strawberry0{j}" for j in [1,2,3]] + \
# [f"tomato{j}" for j in [1]] + \
# [f"apple{j}" for j in [3]] + \
# [f"potato{j}" for j in [3]]

selected_objects = \
[f"lemon0{j}" for j in [2]]

# selected_fruit_names = ["potato"]
# selected_fruit_names = ["apple", "lemon", "potato", "strawberry", "tomato"]
# selected_fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]

# for object_name in object_names:

    # if not any([fruit_name in object_name for fruit_name in selected_fruit_names]):   # if object does NOT belong to any of the selected primitives.
    #     continue

meshes = []
for object_name in selected_objects:


    print(f"object name: {object_name}")
    

    mesh_path = os.path.join(mesh_main_path, object_name, f"{object_name}_processed.stl")
    mesh = trimesh.load(mesh_path)
    # meshes.append(mesh)
    # original_center_mass = mesh.center_mass
    mesh.vertices -= mesh.center_mass   #np.mean(mesh.vertices, axis=0)
    mesh.visual.face_colors = [0, 0, 250, 128]

    # print(mesh.extents)

    hemis_mesh = trimesh.creation.icosphere(radius = 0.028)     # hemi-ellipsoid that matches (almost) exactly with the shape of lemon02  
    hemis_mesh = trimesh.intersections.slice_mesh_plane(mesh=hemis_mesh, plane_normal=[0,-1,0], plane_origin=[0,0.00,0], cap=True)
    hemis_mesh.visual.face_colors = [250, 0, 0, 255]
    hemis_mesh.vertices *= np.array([5./6,1.1,5./6])

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
    # hemis_mesh.vertices -= hemis_mesh.center_mass   # np.mean(hemis_mesh.vertices, axis=0)


    homo_mat_lemon = world_to_object_frame(mesh.vertices)
    homo_mat_hemis = world_to_object_frame(hemis_mesh.vertices)

    transformed_lemon = deepcopy(mesh)
    transformed_lemon.apply_transform(homo_mat_lemon)
    transformed_hemis = deepcopy(hemis_mesh)
    transformed_hemis.apply_transform(homo_mat_hemis)

    transformed_lemon.apply_translation([0.05,0,0])
    transformed_hemis.apply_translation([0.05,0,0])

    trimesh.Scene([mesh, hemis_mesh, transformed_lemon, transformed_hemis]).show()

    hemis_mesh.apply_translation([0.05,0,0])



    print("========================")

    # break

coordinate_frame = trimesh.creation.axis()  
coordinate_frame.apply_scale(0.2)
# trimesh.Scene([mesh, hemis_mesh, coordinate_frame]).show()
# trimesh.Scene(meshes).show()