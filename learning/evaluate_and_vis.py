import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse
import isaacgym
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES
from model import StressNet




static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"
adjacent_tetrahedrals_save_path = "/home/baothach/shape_servo_data/stress_field_prediction/adjacent_tetrahedrals"

data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/data"


start_time = timeit.default_timer() 
visualization = False
query_type = "sampled"  # options: sampled, particle, partial
num_pts = 1024
# stress_visualization_min = 0.001   
# stress_visualization_max = 5e3 
# log_stress_visualization_min = np.log(stress_visualization_min)   
# log_stress_visualization_max = np.log(stress_visualization_max)    

grasp_idx_bounds = [0, 1]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]
    

device = torch.device("cuda")
model = StressNet(num_channels=5).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/weights/run1/epoch 0"))
model.eval()


# for object_name in OBJECT_NAMES:
for object_name in ["strawberry"]:

    get_gripper_pc = True

    # Get adjacent tetrahdras of each vertex
    with open(os.path.join(adjacent_tetrahedrals_save_path, f"{object_name}.pickle"), 'rb') as handle:
        adjacent_tetrahedral_dict = pickle.load(handle)   

    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    partial_pcs = static_data["partial_pcs"][1:2]  # list of 8 point clouds from 8 different camera views
    partial_pcs = [down_sampling(pc, num_pts=num_pts) for pc in partial_pcs]


    for i in range(*grasp_idx_bounds):        
        
        
        for force in force_levels:

            print(f"{object_name} - grasp {i} - force {force}.")

            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{i}_force_{force}.pickle")
            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break 
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)

            (tet_indices, tet_stress) = data["tet"]
            tet_indices = np.array(tet_indices).reshape(-1,4)
            (tri_indices, tri_parents, tri_normals) = data["tri"]
                 
            young_modulus = int(float(data["young_modulus"]))
            full_pc = data["object_particle_state"]
            force = data["force"]
            grasp_pose = data["grasp_pose"]
            fingers_joint_angles = data["fingers_joint_angles"]

            all_stresses = compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0])
            all_stresses_log = np.log(all_stresses)    
            print("min(all_stresses):", min(all_stresses), max(all_stresses)) 
            stress_visualization_min = min(all_stresses)   
            stress_visualization_max = max(all_stresses)
            log_stress_visualization_min = np.log(stress_visualization_min)   
            log_stress_visualization_max = np.log(stress_visualization_max)   

            if get_gripper_pc:
                gripper_pc = get_gripper_point_cloud(grasp_pose, fingers_joint_angles, num_pts=num_pts)
                augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), (gripper_pc.shape[0], 1))))
                pcd_gripper = pcd_ize(gripper_pc, color=[0,0,0])
                get_gripper_pc = False

                augmented_partial_pcs = [np.hstack((pc, np.tile(np.array([force, young_modulus]), (pc.shape[0], 1))))
                                        for pc in partial_pcs]  # list of 8 arrays of shape (num_pts,5)

                # Combine everything together to get an augmented point cloud of shape (num_pts*2,5)
                combined_pcs = [np.concatenate((pc, augmented_gripper_pc), axis=0)
                                for pc in augmented_partial_pcs]  # list of 8 arrays of shape (num_pts + num_pts, 5)

            batch_size = 100
            occupancy_threshold = 0.1   #0.99
            num_query_pts = 2   #1000
            translation = 0.05
            pcds = []
            for j, query_type in enumerate(["sampled", "full"]):
                pcd = reconstruct_stress_field(model, device, batch_size, tri_indices, occupancy_threshold, full_pc, 
                                            combined_pcs[0], query_type, num_query_pts, 
                                            [log_stress_visualization_min, log_stress_visualization_max], return_open3d_object=True)
                pcd.translate((translation*(j+1),0,0))
                pcds.append(pcd)


            pcd_gt = pcd_ize(full_pc)
            # colors = np.array(scalar_to_rgb(all_stresses_log, colormap='jet'))[:,:3]
            colors = np.array(scalar_to_rgb(all_stresses_log, colormap='jet', 
                                            min_val=log_stress_visualization_min, max_val=log_stress_visualization_max))[:,:3]
            pcd_gt.colors = open3d.utility.Vector3dVector(colors)      

            open3d.visualization.draw_geometries(pcds + [pcd_gt, pcd_gripper])


    















