import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb, read_pickle_data
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES
from model import StressNet2



dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/shinghei_data"
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"
gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"


start_time = timeit.default_timer() 
visualization = False
query_type = "sampled"  # options: sampled, full
num_pts = 1024
num_query_pts = 3000
# stress_visualization_min = 0.001   
# stress_visualization_max = 5e3 
# log_stress_visualization_min = np.log(stress_visualization_min)   
# log_stress_visualization_max = np.log(stress_visualization_max)    

grasp_idx_bounds = [0, 1]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]
    

device = torch.device("cuda")
model = StressNet2(num_channels=5).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/run1/epoch 60"))
model.eval()


for i in range(140,141):
    query_data = read_pickle_data(data_path=os.path.join(dataset_path, f"processed sample {i}.pickle"))  # shape (B, 3)
    object_name = query_data["object_name"]
    grasp_idx = query_data["grasp_idx"]
    force = query_data["force"]
    young_modulus = query_data["young_modulus"]   
    gt_stress = query_data["stress_log"]
    print("gt_stress.shape:", gt_stress.shape)

    ### Load robot gripper point cloud
    filtered_data = read_pickle_data(data_path=os.path.join(gripper_pc_path, f"{object_name}_grasp_{grasp_idx}.pickle"))
    gripper_pc = filtered_data["gripper_pc"]
    augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), 
                                    (gripper_pc.shape[0], 1)))) # shape (num_pts,5)
    augmented_gripper_pc = np.tile(augmented_gripper_pc[np.newaxis, :, :], (8, 1, 1)) # shape (8,num_pts,5)
    pcd_gripper = pcd_ize(gripper_pc, color=(0,0,0))

    ### Load partial-view object point clouds
    if "-p1" in object_name or "-p2" in object_name:
        object_name = object_name[:-3]  # Ignore the -p1 and -p2 part.
    partial_pcs = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                    f"{object_name}.pickle"))["partial_pcs"]   # shape (8, num_pts, 3)
    augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                            (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)

    
    ### Combine object pc and gripper pc
    combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1) # shape (8, num_pts*2, 5)
    combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)
    
    
    ### Get query points (sample randomly or use the ground-truth particles)
    if query_type == "sampled":
        full_pc = query_data["query_points"]    #filtered_data["object_particle_states"][0]
        full_pc = full_pc[:round(full_pc.shape[0]/2)]
        print("full_pc.shape:", full_pc.shape)
        query = sample_points_bounding_box(trimesh.PointCloud(full_pc), num_query_pts, scales=[1.5]*3)  # shape (num_query_pts,3) 
    
    # query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
    # query_tensor = query_tensor.unsqueeze(0).repeat(8,1,1).to(device)  # shape (8, num_queries, 3)
    # stress, occupancy = model(combined_pc_tensor, query_tensor) # shape (8*num_queries,1)
    # print(stress.shape, occupancy.shape)
    
    # # stress = stress.view(8, num_query_pts, 1)  # shape (8, num_queries, 1)
    # # occupancy = stress.view(8, num_query_pts, 1)  # shape (8, num_queries, 1)
    # stress = stress.squeeze().cpu().detach().numpy()[:num_query_pts]  # shape (num_queries, 1)
    # occupancy = occupancy.squeeze().cpu().detach().numpy()[:num_query_pts]
    # occupied_idxs = np.where(occupancy >= 0.7)[0]
    
    # print(stress[occupied_idxs].shape)
    
    # pcd = pcd_ize(query[occupied_idxs])
    # pcd.colors = open3d.utility.Vector3dVector(scalar_to_rgb(stress[occupied_idxs])[:,:3])

    pcd_gt = pcd_ize(full_pc)
    pcd_gt.colors = open3d.utility.Vector3dVector(scalar_to_rgb(gt_stress[:round(full_pc.shape[0]/2)])[:,:3])

    # open3d.visualization.draw_geometries([pcd.translate((0.05,0,0)), pcd_gt, pcd_gripper])
    open3d.visualization.draw_geometries([pcd_gt, pcd_gripper])















