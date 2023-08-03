import open3d
import isaacgym
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb, read_pickle_data, print_color
from utils.point_cloud_utils import transform_point_cloud
from utils.stress_utils import *
from utils.camera_utils import grid_layout_images, export_open3d_object_to_image, overlay_texts_on_image, create_media_from_images
from model import StressNet2
from copy import deepcopy


gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
media_path = "/home/baothach/stress_field_prediction/visualization/stress_prediction_results/videos/6polygons"

start_time = timeit.default_timer() 
visualization = False
num_pts = 1024
num_query_pts = 500000
use_open_gripper = True

device = torch.device("cuda")
model = StressNet2(num_channels=5).to(device)
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon/epoch 100"))
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon_open_gripper/epoch 193"))
model.eval()


for idx, file_name in enumerate([f"6polygon0{j}" for j in [2]]):
    object_name = os.path.splitext(file_name)[0]

    print("======================")



    for k in range(0,100):    # 100 grasp poses 32, 7, 8(doesnt work),
        
        gripper_file_name = os.path.join(gripper_pc_recording_path, f"gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")          
        if not os.path.isfile(gripper_file_name):            
            print_color(f"{gripper_file_name} not found")
            continue
        
        with open(gripper_file_name, 'rb') as handle:
            gripper_data = pickle.load(handle)  # gripper when making contact with object
        
        if use_open_gripper:
            file_name = os.path.join(gripper_pc_recording_path, f"open_gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")        
            with open(file_name, 'rb') as handle:
                open_gripper_data = pickle.load(handle) # gripper when maximally open 
            open_gripper_pc = open_gripper_data["transformed_gripper_pcs"][0:1]   # shape (8, num_pts, 3)
            pcd_gripper_open = pcd_ize(open_gripper_pc.squeeze(), color=(1,0,0))
            augmented_gripper_pc_open = np.concatenate([open_gripper_pc, np.tile(np.array([[0, 0]]), 
                                                    (1, open_gripper_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)   
        

        ### Load static data
        static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{object_name}.pickle"))   # shape (8, num_pts, 3)
        adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
        homo_mats = static_data["homo_mats"]
        
        partial_pcs = static_data["transformed_partial_pcs"]
        pc_idx = 0
        partial_pc = partial_pcs[pc_idx:pc_idx+1,:,:]
        partial_pc_ori = static_data["partial_pcs"][pc_idx]

        ### Load robot gripper point cloud when making contact with object
        gripper_pc = gripper_data["transformed_gripper_pcs"][pc_idx:pc_idx+1,:,:]   # shape (8, num_pts, 3)
        # augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
        #                                         (1, gripper_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)      
        pcd_gripper = pcd_ize(gripper_pc.squeeze(), color=(0,0,0))

        ### Sample query points
        query = sample_points_bounding_box(trimesh.PointCloud(partial_pc.squeeze()), num_query_pts, scales=[1.2]*3)  # shape (num_query_pts,3) 



        pcds = []
        pcd_gts = []
        images = []
        vis_gripper = True
        create_media = False

        stress_visualization_min = np.log(1e2)  # 1e3
        stress_visualization_max = np.log(5e4)  # 5e4 1e4

        for force in [0,12]:   # range(0, 16, 3) range(0, 10, 2)
            for young_modulus in [3]:     # [3, 5, 8, 10, 20, 50] [50,20,10,8,5,3]
 
                print(f"{object_name} - grasp {k} - young {young_modulus:.3f} - force {force:.2f} started")
                                  
                ### Augmented object pc
                augmented_partial_pcs = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus]]), 
                                                        (1, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
            
                ### Combine object pc and gripper pc
                if use_open_gripper:
                    combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc_open), axis=1)
                else:
                    combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1) # shape (8, num_pts*2, 5)
                combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)
                
                
                query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
                query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
                stress, occupancy = model(combined_pc_tensor, query_tensor)


                pred_stress = stress.squeeze().cpu().detach().numpy()
                pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
                occupied_idxs = np.where(pred_occupancy >= 0.7)[0]


                pcd = pcd_ize(query[occupied_idxs], color=[0,1,0])
                colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
                pcd.colors = open3d.utility.Vector3dVector(colors)
                
                # # pcd_partial = pcd_ize(partial_pcs.reshape(-1,3), color=[0,0,0])     
                # # open3d.visualization.draw_geometries([pcd, pcd_gripper, pcd_partial.translate((0.07,0.00,0))])

                img_resolution = [1000,1000]
                # cam_position=[0.2, -0.2, 0.15]
                # cam_target = [0, 0, 0]
                # cam_up_vector = [0, 0, 1]
                cam_position=[0.0, 0.0, 1.0]
                cam_target = [0, 0, 0]
                cam_up_vector = [0, 1, 0] 

                if not create_media:            
                    open3d.visualization.draw_geometries([pcd, pcd_gripper, pcd_gripper_open])
                    

                    # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                    # image = export_open3d_object_to_image([pcd + pcd_gripper, coor], None, img_resolution,  
                    #                                       cam_position, cam_target, cam_up_vector, display_on_screen=True)
                
                else:
                    if vis_gripper:
                        image = export_open3d_object_to_image([pcd + pcd_gripper_open], None, img_resolution,  
                                                            cam_position, cam_target, cam_up_vector, display_on_screen=False)
                        # vis_gripper = False
                    else:
                        image = export_open3d_object_to_image([pcd], None, img_resolution,  
                                                            cam_position, cam_target, cam_up_vector, display_on_screen=False)
                    # images.append(image)

                    overlaid_image = overlay_texts_on_image(image, texts=[f"{young_modulus*10:.0f} kPa - {force:.1f} N"], 
                                                            font_size=80, positions=[(200,0)], return_numpy_array=True, display_on_screen=False)
                    images.append(overlaid_image)
                
                
                
                print_color("========================")
                
                # break

        if create_media:
            media_filename = os.path.join(media_path, f"{object_name} - grasp {k}.mp4")

            if os.path.exists(media_filename):
                # Find the first available unique filename
                duplicate_count = 1
                while True:
                    new_filename = f"{os.path.splitext(media_filename)[0]} ({duplicate_count}).mp4"
                    if not os.path.exists(new_filename):
                        break
                    duplicate_count += 1
                media_filename = new_filename

            create_media_from_images(images, output_path=media_filename, frame_duration=2.0, output_format='mp4')



        # break















