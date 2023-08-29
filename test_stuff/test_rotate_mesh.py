import numpy as np
import trimesh
import os
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb, read_pickle_data, print_color
from utils.mesh_utils import simplify_mesh_pymeshlab, create_tet_mesh
import pymeshlab as ml
import open3d as o3d
from copy import deepcopy

def get_mesh_position(idx, num_cols, spacing):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    return [row_idx * spacing, -col_idx * spacing, 0]

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"
recon_mesh_path = "/home/baothach/shape_servo_data/stress_field_prediction/results"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"

# selected_objects = ["mustard_bottle", "strawberry02", "lemon02"]
selected_objects = ["mustard_bottle"]


meshes = []
for i, object_name in enumerate(selected_objects):

    ### Load static data
    static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                    f"{object_name}.pickle"))   # shape (8, num_pts, 3)
    homo_mats = static_data["homo_mats"]

    # file_name = os.path.join(mesh_main_path, f"{object_name}/google_16k/nontextured_proc.stl")
    file_name = os.path.join(mesh_main_path, f"{object_name}/{object_name}_processed.stl")
    
    # mesh = trimesh.load(file_name)
    # Load a mesh from a file (replace 'your_mesh.stl' with the actual filename)
    mesh = o3d.io.read_triangle_mesh(file_name)
    mesh.compute_vertex_normals()   # compute surface normal to enable shading
    # mesh.transform(homo_mats[0])
    mesh.translate(-np.mean(np.asarray(mesh.vertices), axis=0))
    
    file_name = os.path.join(recon_mesh_path, f"{object_name}_recon_mesh.stl")
    mesh_recon = o3d.io.read_triangle_mesh(file_name)
    mesh_recon.compute_vertex_normals()   # compute surface normal to enable shading
    mesh_recon.translate(-np.mean(np.asarray(mesh_recon.vertices), axis=0))
    mesh_recon.translate((0.13,0,0))    # 0.1 0.13

    # Define the rotation axis (for example, [0, 0, 1] for Z-axis)
    rotation_axis = np.array([0, 0, 1])

    # Define the number of frames and rotation speed
    num_frames = 180  # Number of frames for a full rotation
    rotation_speed_deg = 1  # Rotation speed in degrees per frame

    # Convert the rotation speed to radians per frame
    rotation_speed_rad = np.radians(rotation_speed_deg)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the mesh to the visualization
    vis.add_geometry(mesh)
    vis.add_geometry(mesh_recon)


    # Continuous rotation animation loop
    while True:  # Infinite loop for continuous rotation
        for obj in [mesh, mesh_recon]:
            # Calculate the rotation center as the local center of the current object
            rotation_center = obj.get_center()

            # Create a rotation matrix for the current angle
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = np.array([
                [np.cos(rotation_speed_rad), -np.sin(rotation_speed_rad), 0],
                [np.sin(rotation_speed_rad), np.cos(rotation_speed_rad), 0],
                [0, 0, 1]
            ])
            
            # Apply the rotation to the current object around its local center
            obj.translate(-rotation_center)  # Move object to origin
            obj.transform(rotation_matrix)    # Rotate
            obj.translate(rotation_center)    # Move back to center
        
        # Update the visualization
        vis.update_geometry(mesh)
        vis.update_geometry(mesh_recon)
        vis.poll_events()
        vis.update_renderer()






