import numpy as np
import trimesh
import os
import sys
sys.path.append("../")
from utils.mesh_utils import simplify_mesh_pymeshlab, create_tet_mesh
import pymeshlab as ml
from copy import deepcopy

def create_oriented_bounding_box(original_mesh):
    # Compute the oriented bounding box parameters
    obb_transform = original_mesh.bounding_box_oriented.primitive.transform
    obb_extents = original_mesh.bounding_box_oriented.primitive.extents

    # Create the new mesh representing the oriented bounding box
    obb_mesh = trimesh.creation.box(extents=obb_extents, transform=obb_transform)

    return obb_mesh



# def create_oriented_bounding_ellipsoid(original_mesh):
#     # Compute the oriented bounding box parameters
#     obb_transform = original_mesh.bounding_box_oriented.primitive.transform
#     obb_extents = original_mesh.bounding_box_oriented.primitive.extents

#     # Create the new mesh representing the oriented bounding box
#     # obb_mesh = trimesh.creation.box(extents=obb_extents, transform=obb_transform)

#     mesh = trimesh.creation.icosphere(radius = 1)
#     vertices_transformed = mesh.vertices * obb_extents/2.0 
#     obb_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)
#     obb_mesh.apply_transform(obb_transform)

#     return obb_mesh

def create_oriented_bounding_ellipsoid(original_mesh, subdivisions=4, num_points=1000, iterations=20):
    """
    Create an oriented bounding ellipsoid that best encapsulates the input mesh.
    
    Args:
    original_mesh (trimesh.Trimesh): The input mesh.
    subdivisions (int): Number of subdivisions for the initial icosphere.
    num_points (int): Number of points to sample from the ellipsoid for each iteration.
    iterations (int): Number of iterations to refine the ellipsoid mesh.
    
    Returns:
    trimesh.Trimesh: The oriented bounding ellipsoid mesh.
    """
    # Step 1: Compute the bounding box of the original mesh
    bbox = original_mesh.bounding_box.bounds

    # Step 2: Create an initial ellipsoid mesh using bounding box dimensions
    center = np.mean(bbox, axis=0)
    radii = (bbox[1] - bbox[0]) / 2.0

    # Create a sphere (icosphere) with the desired resolution
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)

    # Scale the sphere manually to match the initial ellipsoid
    scaled_vertices = sphere.vertices * radii + center

    # Create a new mesh using the scaled vertices
    ellipsoid_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=sphere.faces)

    # Step 3: Iteratively refine the ellipsoid mesh to fit the original mesh
    for i in range(iterations):
        # Sample points from the ellipsoid
        points = ellipsoid_mesh.sample(num_points)

        # Find the closest points on the original mesh for each sampled point
        closest_points = original_mesh.nearest.on_surface(points)[0]

        # Compute the new center and radii based on the closest points
        new_center = np.mean(closest_points, axis=0)
        new_radii = np.max(np.abs(closest_points - new_center), axis=0)

        # Scale the sphere manually based on the new radii
        scaled_vertices = sphere.vertices * new_radii + new_center

        # Create a new mesh using the scaled vertices
        ellipsoid_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=sphere.faces)

    return ellipsoid_mesh



def get_mesh_position(idx, num_cols, spacing):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    return [row_idx * spacing, -col_idx * spacing, 0]

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"

# selected_objects = ["strawberry02", "lemon02", "mustard_bottle"]
selected_objects = ["lemon02"]


scene = trimesh.Scene()   

for i, object_name in enumerate(selected_objects):
    # file_name = os.path.join(mesh_main_path, f"{object_name}/google_16k/nontextured_proc.stl")
    file_name = os.path.join(mesh_main_path, f"{object_name}/{object_name}_processed.stl")
    
    mesh = trimesh.load(file_name)
    # mesh.show()
    print(mesh.extents)

    # scene.adds_geometry(mesh.apply_translation(get_mesh_position(i, num_cols=2, spacing=0.1)))
    
    scene.add_geometry(mesh)
    
    if object_name == "mustard_bottle":
        obb_mesh = create_oriented_bounding_box(mesh)
        new_obj_name = "box08"
        
    elif object_name == "strawberry02":
        obb_mesh = create_oriented_bounding_ellipsoid(mesh)
        obb_mesh = simplify_mesh_pymeshlab(obb_mesh, target_num_vertices=100)
        new_obj_name = "ellipsoid05"
        
    elif object_name == "lemon02":
        obb_mesh = trimesh.convex.convex_hull(mesh.vertices)
        # print(obb_mesh.vertices.shape)
        obb_mesh = simplify_mesh_pymeshlab(obb_mesh, target_num_vertices=10)
        new_obj_name = "box09"    
        
    
    # Set the transparency (alpha) value for the obb_mesh
    obb_mesh.visual.face_colors = [255.0, 0.0, 0.0, 128]
    scene.add_geometry(deepcopy(obb_mesh.apply_translation([0.05,0,0])))
    
    os.makedirs(os.path.join(mesh_main_path, f"{new_obj_name}"), exist_ok=True)
    save_mesh_fname = os.path.join(mesh_main_path, f"{new_obj_name}/{new_obj_name}.stl")
    # obb_mesh.export(save_mesh_fname)    
    # create_tet_mesh(os.path.join(mesh_main_path, f"{new_obj_name}"), new_obj_name, coarsen=True, verbose=True)
    
    # break

coordinate_frame = trimesh.creation.axis()  
coordinate_frame.apply_scale(0.1)
scene.add_geometry(coordinate_frame)  
scene.show()