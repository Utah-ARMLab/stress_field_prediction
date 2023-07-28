import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../")

from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data, sample_points_from_tet_mesh, print_color



""" 
fix object_height_buffer problem 0.001 in 6polygon04 data.
"""

static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"

data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
data_recording_path = os.path.join(data_main_path, "all_6polygon_data")

start_time = timeit.default_timer() 
visualization = False
fix_data = False

grasp_idx_bounds = [0, 100]


for object_name in [f"6polygon0{j}" for j in [4]]:


    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)
    partial_pcs = static_data["partial_pcs"]  # shape (8, num_pts, 3)
    partial_pc = partial_pcs.reshape(-1,3)
    

    for grasp_idx in range(*grasp_idx_bounds):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in range(0,61):
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                # print_color(f"{file_name} not found")
                break   
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)

            print("(BEFORE) max_z partial + full:", max(partial_pc[:,2]), max(data["object_particle_state"][:,2]))
            pcd_partial = pcd_ize(partial_pc, color=[0,0,1])
            pcd_full = pcd_ize(data["object_particle_state"], color=[1,0,0])
            open3d.visualization.draw_geometries([pcd_full, pcd_partial])
            
            
            data["object_particle_state"][:,2] -= 0.001
            print("(AFTER) max_z partial + full:", max(partial_pc[:,2]), max(data["object_particle_state"][:,2]))
            pcd_full = pcd_ize(data["object_particle_state"], color=[1,0,0])
            open3d.visualization.draw_geometries([pcd_full, pcd_partial])  
            
            if fix_data:            
                with open(file_name, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                    
                    
            break
        break                       