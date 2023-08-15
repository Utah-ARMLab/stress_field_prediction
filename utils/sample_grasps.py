import numpy as np
import trimesh
import os
import sys
sys.path.append("../")
from utils.grasp_utils import sample_grasps
np.random.seed(2023)

mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness"

# selected_objects = ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol", "mustard_bottle"]
selected_objects = ["box08", "box09", "ellipsoid05"]    #"mustard_bottle"

meshes = []
for i, object_name in enumerate(selected_objects):
    
    file_name = os.path.join(mesh_main_path, f"{object_name}/{object_name}.stl")
    mesh = trimesh.load(file_name)
    
    save_grasps_dir = os.path.join(mesh_main_path, f"{object_name}", f"{object_name}_grasps.h5")
    grasps = sample_grasps(mesh, cls_sampler="AntipodalSampler",
                           number_of_grasps=100, visualization=True, vis_gripper_name='panda_tube',
                           save_grasps_dir=save_grasps_dir)
    
    # break