import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import random
import isaacgym
import torch
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


import sys
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb, read_pickle_data, print_color, down_sampling
from utils.point_cloud_utils import transform_point_cloud, is_homogeneous_matrix
from utils.stress_utils import *
from utils.camera_utils import grid_layout_images, export_open3d_object_to_image, overlay_texts_on_image, create_media_from_images
from utils.mesh_utils import sample_points_from_tet_mesh
from model import StressNet2, PointCloudEncoder
from copy import deepcopy
import re


# gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/all_primitives"
gripper_pc_recording_path = os.path.join(data_main_path,  f"processed")

media_path = "/home/baothach/stress_field_prediction/visualization/stress_prediction_results/videos/all_primitives/"
os.makedirs(media_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = False
num_pts = 1024
num_query_pts = 500000
use_open_gripper = True
joint_training = False

grasp_idx_bounds = [0, 100]


device = torch.device("cuda")
# model = StressNet2(num_channels=5, joint_training=joint_training).to(device)
model = StressNet2(num_channels=5, pc_encoder_type=PointCloudEncoder, joint_training=joint_training).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/weights/mesh_cylinders_closed_gripper_partial/epoch 57"))
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon/epoch 100"))
model.eval()


fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
# selected_primitive_names = ["ellipsoid"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]

selected_objects = []
# selected_objects += \
# [f"lemon0{j}" for j in [1,2,3]] + \
# [f"strawberry0{j}" for j in [1,2,3]] + \
# [f"tomato{j}" for j in [1]] + \
# [f"apple{j}" for j in [3]] + \
# [f"potato{j}" for j in [3]]
# selected_objects += ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol"]
# selected_objects += [f"cylinder0{j}" for j in range(1,9)] + [f"box0{j}" for j in range(1,10)] \
#                 + [f"ellipsoid0{j}" for j in range(1,6)] + [f"sphere0{j}" for j in [1,3,4,6]]

# "mustard_bottle" lemon02 strawberry02

selected_objects += [f"cylinder06"]

for idx, file_name in enumerate(selected_objects):
    object_name = os.path.splitext(file_name)[0]
    prim_name = object_name[:-2]    #re.search(r'(\D+)', object_name).group(1)
    data_recording_path = os.path.join(data_main_path, f"all_{prim_name}_data")

    ### Load static data
    static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                    f"{object_name}.pickle"))   # shape (8, num_pts, 3)
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    tet_indices = static_data["tet_indices"]
    
    # mesh = static_data["downsampled_mesh_vertices"]
    mesh = static_data["transformed_partial_pcs"][0]


    print("======================")

    images = []
    for k in range(0,100):    # 100 grasp poses 32, 7, 8(doesnt work),

        
        # file_name = os.path.join(gripper_pc_recording_path, f"open_gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")        
        # if not os.path.isfile(file_name):
            
        #     print_color(f"{file_name} not found")
        #     continue
        # with open(file_name, 'rb') as handle:
        #     gripper_data = pickle.load(handle)

        if use_open_gripper:
            file_name = os.path.join(gripper_pc_recording_path, f"gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")        
            with open(file_name, 'rb') as handle:
                open_gripper_data = pickle.load(handle)
            open_gripper_pc = open_gripper_data["gripper_pc"]   # shape (8, num_pts, 3)
            # open_gripper_pc = open_gripper_data["gripper_pc"][np.newaxis, :]
            pcd_gripper_open = pcd_ize(open_gripper_pc.squeeze(), color=(1,0,0))
            augmented_gripper_pc_open = np.concatenate([open_gripper_pc, np.tile(np.array([[0, 0]]), 
                                                        (open_gripper_pc.shape[0], 1))], axis=1)   # shape (num_pts, 5)  


        ### Sample query points
        query = sample_points_bounding_box(trimesh.PointCloud(mesh), num_query_pts, scales=[1.5]*3)  # shape (num_query_pts,3) 
        # query = partial_pc.squeeze()

        pcds = []
        pcd_gts = []
        
        vis_gripper = True
        create_media = False

        stress_visualization_min = np.log(1e2)  # 1e3
        stress_visualization_max = np.log(1e5)  # 5e4 1e4

        for force_idx in [0]:
        # for force_idx in range(0,25,5):
            
            print(f"{object_name} - grasp {k} - force {force_idx} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{k}_force_{force_idx}.pickle")

            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break 

            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                
            
            full_pc = data["object_particle_state"]
            force = data["force"] 
            # force = 7    
            print(f"force: {force:.2f}") 
            

            young_modulus = float(data["young_modulus"])/1e4
            # young_modulus = 2
            print(f"young_modulus: {young_modulus:.3f}")
            
            tet_stress = data["tet_stress"]
            gt_stress = np.log(compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0]))

            # full_pc_w_stress = np.concatenate((full_pc, gt_stress[:, np.newaxis]), axis=1)      
            # points_from_tet_mesh = sample_points_from_tet_mesh(full_pc_w_stress[tet_indices], k=4)   # sample points from the volumetric mesh
            # print(full_pc.shape, points_from_tet_mesh.shape)
            # query_points_volume = np.concatenate((full_pc, points_from_tet_mesh[:,:3])) # concat the object particles with these newly selected points.
            # stress_volume = np.concatenate((full_pc_w_stress[:,3:], points_from_tet_mesh[:,3:])).squeeze()   # stress at each query points            

            
            augmented_mesh = np.concatenate([mesh, np.tile(np.array([[force, young_modulus]]), 
                                            (mesh.shape[0], 1))], axis=1)   # shape (num_pcs, num_pts, 5)
        
            ### Combine object pc and gripper pc            
            if use_open_gripper:
                combined_pcs = np.concatenate((augmented_mesh, augmented_gripper_pc_open), axis=0)
            else:
                combined_pcs = np.concatenate((augmented_mesh, augmented_gripper_pc), axis=1) # shape (num_pts*2, 5)
            
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(1,0).float().unsqueeze(0).to(device)  # shape (1, 5, num_pts*2)
            print(augmented_mesh.shape, augmented_gripper_pc_open.shape, combined_pcs.shape)
            
            query_tensor = torch.from_numpy(query).float()  # shape (num_queries, 3)
            query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)


            if joint_training:
                stress, occupancy = model(combined_pc_tensor, query_tensor)
                pred_stress = stress.squeeze().cpu().detach().numpy()
            else:
                occupancy = model(combined_pc_tensor, query_tensor)
                pred_stress = np.ones(query.shape[0]) * stress_visualization_min
           
            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.7)[0]


            pcd_gt = pcd_ize(full_pc, color=[1,0,0])
            colors = np.array(scalar_to_rgb(gt_stress, colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            pcd_gt.colors = open3d.utility.Vector3dVector(colors)


            pcd = pcd_ize(query[occupied_idxs], color=[0,1,0])
            colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            pcd.colors = open3d.utility.Vector3dVector(colors)
           
            pcd_query = pcd_ize(query, color=[0,1,0]) 
            pcd_mesh = pcd_ize(mesh, color=[0,0,1])

            if not create_media:            

                if use_open_gripper:

                    translation = (-0.1,0.00,0)   
                    temp_pcd_gripper_open = deepcopy(pcd_gripper_open).translate(translation)
                    temp_pcd_gripper_open.paint_uniform_color([0,0,1])
                    pcd.paint_uniform_color([0,0,0])
                    pcd_gt.paint_uniform_color([1,0,0]) #pcd_partial.translate(translation)

 
                    open3d.visualization.draw_geometries([pcd.translate(translation), temp_pcd_gripper_open, 
                                                        pcd_gt, pcd_gripper_open, pcd_mesh])   

           
               
            print_color("========================")
            
            break

        # break















