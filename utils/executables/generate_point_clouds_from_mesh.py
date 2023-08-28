import trimesh
import os
import numpy as np
import sys
sys.path.append("../../")
from utils.mesh_utils import generate_point_clouds_from_mesh
from utils.miscellaneous_utils import pcd_ize

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"
# object_names = [f"hemi0{j}" for j in [1]] + [f"ellipsoid0{j}" for j in [1,2]] + \
#                     [f"sphere0{j}" for j in [3,4,6]]
object_names = ["lemon02"]

meshes = []

for i, obj_name in enumerate(object_names):
    mesh_dir = os.path.join(mesh_main_path, obj_name)
    fname_object = os.path.join(mesh_dir, f"{obj_name}.stl")
    mesh = trimesh.load(fname_object)
    
    generated_pcs = generate_point_clouds_from_mesh(mesh, obj_name, num_poses_per_obj=1)
    print(generated_pcs)
    pcd_ize(generated_pcs[0], color=[0,0,0], vis=True)


# trimesh.Scene(meshes).show()


