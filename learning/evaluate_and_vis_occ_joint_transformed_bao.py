import open3d
import isaacgym
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb, read_pickle_data, print_color
from utils.point_cloud_utils import transform_point_cloud
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES
from model import StressNet2
from copy import deepcopy


gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/gripper_data_6polygon04"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
# dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_6polygon04"
data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/all_6polygon_data"
data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/all_6polygon_data"

start_time = timeit.default_timer() 
visualization = False
query_type = "sampled"  # options: sampled, full
num_pts = 1024
num_query_pts = 100000
# stress_visualization_min = 0.001   
# stress_visualization_max = 5e3 
# log_stress_visualization_min = np.log(stress_visualization_min)   
# log_stress_visualization_max = np.log(stress_visualization_max)    

grasp_idx_bounds = [0, 100]
# force_levels = np.arange(0.0, 15.25, 2.0)  #np.arange(1, 15.25, 0.25)    [1.0]
# force_levels = [0.0,3.0,8.0,11.25]   

device = torch.device("cuda")
model = StressNet2(num_channels=5).to(device)
# model = StressNetSDF(num_channels=5).to(device)
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/6polygon04/epoch 100"))
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/6polygon04_exp_2/epoch 400"))
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/weights/all_6polygon_joint_transformed/epoch 69"))
model.eval()


fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
# selected_primitive_names = ["ellipsoid"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]


for idx, file_name in enumerate([f"6polygon0{j}" for j in [4]]):
    object_name = os.path.splitext(file_name)[0]

    print("======================")
    print_color(f"{object_name}, {idx}")
    print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")


    for k in range(8,100):    # 100 grasp poses 32, 7, 8(doesnt work),
        
        file_name = os.path.join(gripper_pc_recording_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            
            print_color(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            gripper_data = pickle.load(handle)
        
        

        ### Load static data
        static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{object_name}.pickle"))   # shape (8, num_pts, 3)
        adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
        homo_mats = static_data["homo_mats"]
        
        partial_pcs = static_data["transformed_partial_pcs"]
        pc_idx = 0
        partial_pc = partial_pcs[pc_idx:pc_idx+1,:,:]
        partial_pc_ori = static_data["partial_pcs"][pc_idx]

        ### Load robot gripper point cloud
        gripper_pc = gripper_data["transformed_gripper_pcs"][pc_idx:pc_idx+1,:,:]   # shape (8, num_pts, 3)
        augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
                                                (1, gripper_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)      
        pcd_gripper = pcd_ize(gripper_pc.squeeze(), color=(0,0,0))

        ### Sample query points
        query = sample_points_bounding_box(trimesh.PointCloud(partial_pc.squeeze()), num_query_pts, scales=[1.2]*3)  # shape (num_query_pts,3) 


        pcds = []
        pcd_gts = []

        for force_idx in [30]:
        # for force_idx in range(20,61):
            
            print(f"{object_name} - grasp {k} - force {force_idx} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{k}_force_{force_idx}.pickle")

            if not os.path.isfile(file_name):
                print(f"{file_name} not found")
                break 

            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
                
            
            full_pc = data["object_particle_state"]
            force = data["force"] 
            force = 7    
            print(f"force: {force:.2f}") 
            
            # young_modulus = np.log(float(data["young_modulus"]))
            # # print(float(data["young_modulus"]))
            # # young_modulus = np.log(7e4)
            # print(np.log(1e5), np.log(5e4))
            # print(f"young_modulus: {np.exp(young_modulus)/1e4:.3f}e4")

            young_modulus = float(data["young_modulus"])/1e4
            # print(float(data["young_modulus"]))
            young_modulus = 2
            print(f"young_modulus: {young_modulus:.3f}")
            
            tet_stress = data["tet_stress"]
            gt_stress = np.log(compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0]))
                  



            augmented_partial_pcs = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus]]), 
                                                    (1, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
        
            ### Combine object pc and gripper pc
            combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1) # shape (8, num_pts*2, 5)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)
            
            
            query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
            query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
            stress, occupancy = model(combined_pc_tensor, query_tensor)


            pred_stress = stress.squeeze().cpu().detach().numpy()
            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.7)[0]

            pcd_gt = pcd_ize(transform_point_cloud(full_pc, homo_mats[pc_idx]), color=[1,0,0])  # transform ground truth full_pc from world frame to object frame.
            colors = np.array(scalar_to_rgb(gt_stress, colormap='jet'))[:,:3]
            pcd_gt.colors = open3d.utility.Vector3dVector(colors)


            pcd = pcd_ize(query[occupied_idxs], color=[0,1,0])
            colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=min(gt_stress), max_val=max(gt_stress)))[:,:3]
            pcd.colors = open3d.utility.Vector3dVector(colors)


            

            # open3d.visualization.draw_geometries([pcd.translate((-0.12,0.00,0)), deepcopy(pcd_gripper).translate((-0.12,0.00,0)), pcd_gt, pcd_gripper])
            
            print_color("========================")

            pcds.append(pcd)
            pcd_gts.append(pcd_gt)

            vis = open3d.visualization.Visualizer()
            vis.create_window(visible=True, width=600, height=600) #works for me with False, on some systems needs to be true
            vis.add_geometry(pcd)
            # vis.update_geometry(pcd)

            # Get the view control
            view_control = vis.get_view_control()

            # Set the default view control parameters
            front = [0.0, 0.0, 1.0]  # Camera direction (default is [0, 0, -1])
            lookat = [0.0, 0.0, 0.0]  # Camera target point (default is [0, 0, 0])
            view_control.set_front(front)
            view_control.set_lookat(lookat)


            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image("/home/baothach/Downloads/point_cloud_image.png")
            vis.destroy_window()
            
            # # Create a visualization object and add the point cloud
            # vis = open3d.visualization.Visualizer()
            # vis.create_window()
            # vis.add_geometry(pcd)

            # # Capture the rendered image and save it to a file
            # image = vis.capture_screen_float_buffer()
            # open3d.io.write_image("/home/baothach/Downloads/point_cloud_image.png", image)       
           
            # # Close the visualization window
            # vis.destroy_window()
                       
            
            # break

        # for i, pcd in enumerate(pcds):
        #     pcd.translate((0.0,0.06*(i),0))

        # for i, pcd_gt in enumerate(pcd_gts):
        #     pcd_gt.translate((0.04,0.06*(i),0))

        # open3d.visualization.draw_geometries(pcds + pcd_gts)  # + pcd_gts

        break















