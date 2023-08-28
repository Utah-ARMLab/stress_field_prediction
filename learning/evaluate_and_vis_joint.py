import open3d
import isaacgym
import os
import numpy as np
import pickle5 as pickle
import timeit
import sys
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, scalar_to_rgb, read_pickle_data, print_color
from utils.point_cloud_utils import transform_point_cloud, is_homogeneous_matrix
from utils.stress_utils import *
from utils.camera_utils import grid_layout_images, export_open3d_object_to_image, overlay_texts_on_image, create_media_from_images
from utils.mesh_utils import sample_points_from_tet_mesh
from model import StressNet2
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
num_query_pts = 50000
use_open_gripper = True

grasp_idx_bounds = [0, 100]


device = torch.device("cuda")
model = StressNet2(num_channels=5).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/weights/all_object_occ/epoch 100"))
model.eval()


fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
# selected_primitive_names = ["ellipsoid"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]


for idx, file_name in enumerate(["cylinder04"]):
    object_name = os.path.splitext(file_name)[0]
    prim_name = re.search(r'(\D+)', object_name).group(1)
    data_recording_path = os.path.join(data_main_path, f"all_{prim_name}_data")
    

    print("======================")

    images = []
    for k in range(17,100):    # 100 grasp poses 32, 7, 8(doesnt work),

        
        file_name = os.path.join(gripper_pc_recording_path, f"open_gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            
            print_color(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            gripper_data = pickle.load(handle)

        if use_open_gripper:
            file_name = os.path.join(gripper_pc_recording_path, f"open_gripper_data_{object_name}", f"{object_name}_grasp_{k}.pickle")        
            with open(file_name, 'rb') as handle:
                open_gripper_data = pickle.load(handle)
            # open_gripper_pc = open_gripper_data["transformed_gripper_pcs"][0:1]   # shape (8, num_pts, 3)
            open_gripper_pc = open_gripper_data["gripper_pc"][np.newaxis, :]
            pcd_gripper_open = pcd_ize(open_gripper_pc.squeeze(), color=(1,0,0))
            augmented_gripper_pc_open = np.concatenate([open_gripper_pc, np.tile(np.array([[0, 0]]), 
                                                    (1, open_gripper_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)      

        

        ### Load static data
        static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{object_name}.pickle"))   # shape (8, num_pts, 3)
        adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
        homo_mats = static_data["homo_mats"]
        tet_indices = static_data["tet_indices"]
        
        partial_pcs = static_data["transformed_partial_pcs"]
        pc_idx = 0
        partial_pc = partial_pcs[pc_idx:pc_idx+1,:,:]
        # partial_pc_ori = static_data["partial_pcs"][pc_idx]

        ### Load robot gripper point cloud
        # gripper_pc = gripper_data["transformed_gripper_pcs"][pc_idx:pc_idx+1,:,:]   # shape (8, num_pts, 3)
        gripper_pc = gripper_data["gripper_pc"][np.newaxis, :]
        augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
                                                (1, gripper_pc.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)      
        pcd_gripper = pcd_ize(gripper_pc.squeeze(), color=(0,0,0))

        ### Sample query points
        query = sample_points_bounding_box(trimesh.PointCloud(partial_pc.squeeze()), num_query_pts, scales=[3]*3)  # shape (num_query_pts,3) 


        pcds = []
        pcd_gts = []
        
        vis_gripper = True
        create_media = False

        stress_visualization_min = np.log(1e2)  # 1e3
        stress_visualization_max = np.log(5e4)  # 5e4 1e4

        for force_idx in [30]:
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

            full_pc_w_stress = np.concatenate((full_pc, gt_stress[:, np.newaxis]), axis=1)      
            # points_from_tet_mesh = sample_points_from_tet_mesh(full_pc_w_stress[tet_indices], k=4)   # sample points from the volumetric mesh
            # query_points_volume = np.concatenate((full_pc, points_from_tet_mesh[:,:3])) # concat the object particles with these newly selected points.
            # stress_volume = np.concatenate((full_pc_w_stress[:,3:], points_from_tet_mesh[:,3:])).squeeze()   # stress at each query points            


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

            # pcd_gt = pcd_ize(transform_point_cloud(full_pc, homo_mats[pc_idx]), color=[1,0,0])  # transform ground truth full_pc from world frame to object frame.
            # colors = np.array(scalar_to_rgb(gt_stress, colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            
            # pcd_gt = pcd_ize(transform_point_cloud(query_points_volume, homo_mats[pc_idx]), color=[1,0,0])  # transform ground truth full_pc from world frame to object frame.           
            # colors = np.array(scalar_to_rgb(stress_volume, colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            pcd_gt = pcd_ize(transform_point_cloud(full_pc, homo_mats[pc_idx]), color=[1,0,0])
            # pcd_gt = pcd_ize(full_pc, color=[1,0,0])
            colors = np.array(scalar_to_rgb(gt_stress, colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            pcd_gt.colors = open3d.utility.Vector3dVector(colors)

            assert is_homogeneous_matrix(homo_mats[pc_idx])

            pcd_partial = pcd_ize(partial_pc.squeeze(), color=(0,1,0))
            # open_gripper_pc = transform_point_cloud(open_gripper_data["gripper_pc"], homo_mats[pc_idx])
            open_gripper_pc = open_gripper_data["transformed_gripper_pcs"][0]
            pcd_gripper_open = pcd_ize(open_gripper_pc, color=(1,0,0))


            pcd = pcd_ize(query[occupied_idxs], color=[0,1,0])
            # colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=min(gt_stress), max_val=max(gt_stress)))[:,:3]
            colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=stress_visualization_min, max_val=stress_visualization_max))[:,:3]
            pcd.colors = open3d.utility.Vector3dVector(colors)
           

            

            img_resolution = [1000,1000]
            zoom = 1#.2
            
            # cam_position=[0,0,1]
            # cam_target = [0, 0, 0]
            # cam_up_vector = [0, 1, 0]   # [0, 0, 1]

            # cam_position=[0.1, 0.15, 0.1]
            # cam_target = [0, 0, 0]
            # cam_up_vector = [0, 0, 1]

            cam_position=[0.2, -0.2, 0.15]
            cam_target = [0, 0, 0]
            cam_up_vector = [0, 0, 1]

            if not create_media:            
                # open3d.visualization.draw_geometries([pcd, pcd_gripper])
                # open3d.visualization.draw_geometries([pcd.translate((-0.12,0.00,0)), deepcopy(pcd_gripper).translate((-0.12,0.00,0)), pcd_gt, pcd_gripper]) 
                if use_open_gripper:
                    # open3d.visualization.draw_geometries([pcd.translate((-0.12,0.00,0)), deepcopy(pcd_gripper).translate((-0.12,0.00,0)), 
                    #                                     deepcopy(pcd_gripper_open).translate((-0.12,0.00,0)), 
                    #                                     pcd_gt, pcd_gripper, pcd_gripper_open])   
                
                    translation = (-0.00,0.00,0)   
                    temp_pcd_gripper_open = deepcopy(pcd_gripper_open).translate(translation)
                    temp_pcd_gripper_open.paint_uniform_color([0,0,1])
                    pcd.paint_uniform_color([0,0,0])
                    pcd_gt.paint_uniform_color([1,0,0])

                    # open3d.visualization.draw_geometries([pcd.translate(translation), deepcopy(pcd_gripper_open).translate(translation), 
                    #                                     temp_pcd_gripper_open, 
                    #                                     pcd_gt, pcd_gripper_open])   

                    open3d.visualization.draw_geometries([pcd.translate(translation), deepcopy(pcd_gripper_open).translate(translation), 
                                                        temp_pcd_gripper_open, 
                                                        pcd_gt, pcd_gripper_open])   

                # # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                # # image = export_open3d_object_to_image([pcd + pcd_gripper, coor], None, img_resolution,  
                # #                                         cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                # # image_gt = export_open3d_object_to_image([pcd_gt + pcd_gripper, coor], None, img_resolution,  
                # #                                         cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                # grid_layout_images([image, image_gt], num_columns=2, display_on_screen=True)

            else:
                if vis_gripper:
                    image = export_open3d_object_to_image([pcd + pcd_gripper], None, img_resolution,  
                                                        cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                    image_gt = export_open3d_object_to_image([pcd_gt + pcd_gripper], None, img_resolution,  
                                                        cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                    combined_image = grid_layout_images([image, image_gt], num_columns=2, display_on_screen=False)
                else:
                    image_gt = export_open3d_object_to_image([pcd], None, img_resolution,  
                                                        cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                    image_gt = export_open3d_object_to_image([pcd_gt], None, img_resolution,  
                                                        cam_position, cam_target, cam_up_vector, zoom, display_on_screen=False)
                    combined_image = grid_layout_images([image, image_gt], num_columns=2, display_on_screen=False)


                overlaid_image = overlay_texts_on_image(combined_image, texts=[f"{young_modulus*10:.0f} kPa - {force:.1f} N"], 
                                                        font_size=80, positions=[(700,0)], return_numpy_array=True, display_on_screen=False)
                images.append(overlaid_image)

               
            print_color("========================")
            
            break

        # break

    if create_media:
        media_filename = os.path.join(media_path, f"{object_name}.mp4")

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













