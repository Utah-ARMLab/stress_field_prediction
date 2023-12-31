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
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES
from model import StressNet2


gripper_pc_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/gripper_data_6polygon04"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
# dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_6polygon04"
data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/all_6polygon_data_new"

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
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/weights/6polygon04_8pc_joint/epoch 150"))
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


    for k in range(0,100):    # 100 grasp poses
        
        file_name = os.path.join(gripper_pc_recording_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            
            print_color(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            gripper_data = pickle.load(handle)
        
        
        ### Load robot gripper point cloud
        gripper_pc = gripper_data["gripper_pc"]     
        augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), 
                                        (gripper_pc.shape[0], 1)))) # shape (num_pts,5)
        augmented_gripper_pc = np.tile(augmented_gripper_pc[np.newaxis, :, :], (8, 1, 1)) # shape (8,num_pts,5)
        pcd_gripper = pcd_ize(gripper_pc, color=(0,0,0))



        ### Load partial-view object point clouds
        static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{object_name}.pickle"))   # shape (8, num_pts, 3)
        partial_pcs = static_data["partial_pcs"]
        adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]



        file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{k}_force_{0}.pickle")        
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
            query = sample_points_bounding_box(trimesh.PointCloud(data["object_particle_state"]), num_query_pts, scales=[1.5,1.5,1.5])  # shape (num_query_pts,3) 
            sample_query = False


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
                
            young_modulus = np.log(float(data["young_modulus"]))
            full_pc = data["object_particle_state"]
            force = data["force"]     
            print("force:", force) 
            tet_stress = data["tet_stress"]
            gt_stress = np.log(compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0]))
                  



            augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                    (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
        
            ### Combine object pc and gripper pc
            combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1)[3:4,:,:] # shape (8, num_pts*2, 5)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)
            
            
            query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
            query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
            stress, occupancy = model(combined_pc_tensor, query_tensor)


            pred_stress = stress.squeeze().cpu().detach().numpy()
            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.7)[0]

            pcd_gt = pcd_ize(full_pc, color=[0,0,0])
            colors = np.array(scalar_to_rgb(gt_stress, colormap='jet'))[:,:3]
            pcd_gt.colors = open3d.utility.Vector3dVector(colors)


            pcd = pcd_ize(query[occupied_idxs])
            colors = np.array(scalar_to_rgb(pred_stress[occupied_idxs], colormap='jet', min_val=min(gt_stress), max_val=max(gt_stress)))[:,:3]
            pcd.colors = open3d.utility.Vector3dVector(colors)


            

            open3d.visualization.draw_geometries([pcd.translate((-0.09,0.00,0)), pcd_gt, pcd_gripper])
            
            print_color("========================")

            pcds.append(pcd)
            pcd_gts.append(pcd_gt)
                       
            
            # break

        # for i, pcd in enumerate(pcds):
        #     pcd.translate((0.0,0.06*(i),0))

        # for i, pcd_gt in enumerate(pcd_gts):
        #     pcd_gt.translate((0.04,0.06*(i),0))

        # open3d.visualization.draw_geometries(pcds + pcd_gts)  # + pcd_gts

        # break















