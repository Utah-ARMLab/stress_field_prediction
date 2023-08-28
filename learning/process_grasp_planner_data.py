import open3d
#import isaacgym
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
from utils.graspsampling_utilities import poses_wxyz_to_mats
#from utils.camera_utils import grid_layout_images, export_open3d_object_to_image, overlay_texts_on_image, create_media_from_images
import torchvision
from torchvision.utils import make_grid
#####################################################################
from model import StressNet2
from copy import deepcopy
from utils.grasp_utils import sample_grasps
from grasp_planner import pose_to_homo_mat
import trimesh.transformations as tra
import argparse
from loop_grasp_planner import STATIC_DATA_PATH

'''
Prepare training data for grasp planner
'''

def get_gripper_point_cloud_no_transform(fingers_joint_angles, num_pts=1024, gripper_name='panda', finger_only=True):
    gripper = create_gripper(gripper_name, configuration=fingers_joint_angles, 
                             franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=finger_only)
    mesh = gripper.mesh.copy()
    return trimesh.sample.sample_surface_even(mesh, count=num_pts)[0]



def sample_grid_points_bounding_box(object, scales=[1.5, 1.5, 1.5], voxel_size=0.1, seed=None):
    """ 
    Sample points from the bounding box of the object mesh.
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(object, trimesh.Trimesh):
        # Get the bounding box of the mesh
        bbox = object.bounding_box

        # Get the minimum and maximum coordinates of the bounding box
        min_coords = bbox.bounds[0]
        max_coords = bbox.bounds[1]
    
    elif isinstance(object, trimesh.PointCloud):
        min_coords = object.bounds[0]
        max_coords = object.bounds[1]        

    # Calculate the dimensions of the bounding box
    dimensions = max_coords - min_coords

    # Extend the dimensions by a factor of 'scales'
    extended_dimensions = dimensions * np.array(scales)

    # Calculate the center of the extended box
    center = (min_coords + max_coords) / 2.0

    # Calculate the minimum and maximum coordinates of the extended box
    extended_min_coords = center - extended_dimensions / 2.0
    extended_max_coords = center + extended_dimensions / 2.0

    # print("min: ", extended_min_coords)
    # print("max: ", extended_max_coords)
    lengths = extended_max_coords - extended_min_coords
    # voxel_sizes = [voxel_size, voxel_size, lengths[2]/lengths[0]*voxel_size]
    voxel_sizes = [voxel_size, voxel_size, voxel_size]
    points = voxelize_and_calculate_centroids(extended_min_coords, extended_max_coords, voxel_sizes)

    return points

def voxelize_and_calculate_centroids(min_coords, max_coords, voxel_sizes):
    # Calculate the number of voxels along each axis
    voxel_size_x, voxel_size_y, voxel_size_z = voxel_sizes
    print("x:", voxel_size_x)
    num_voxels_x = int((max_coords[0] - min_coords[0]) // voxel_size_x + 1)
    num_voxels_y = int((max_coords[1] - min_coords[1]) // voxel_size_y + 1)
    num_voxels_z = int((max_coords[2] - min_coords[2]) // voxel_size_z + 1)

    # Create a 3D array to represent the voxel grid
    voxel_grid = [[[False for _ in range(num_voxels_z)] for _ in range(num_voxels_y)] for _ in range(num_voxels_x)]

    # Iterate through each voxel's position in the grid
    for x in range(num_voxels_x):
        for y in range(num_voxels_y):
            for z in range(num_voxels_z):
                voxel_center = (
                    min_coords[0] + x * voxel_size_x + voxel_size_x / 2,
                    min_coords[1] + y * voxel_size_y + voxel_size_y / 2,
                    min_coords[2] + z * voxel_size_z + voxel_size_z / 2
                )
                is_in_bounding_box = (voxel_center[0]<=max_coords[0]) and (voxel_center[1]<=max_coords[1]) and (voxel_center[2]<=max_coords[2])
                if is_in_bounding_box:
                    voxel_grid[x][y][z] = True


    centroids = []
    # Calculate centroids of occupied voxels
    for x in range(num_voxels_x):
        for y in range(num_voxels_y):
            for z in range(num_voxels_z):
                if voxel_grid[x][y][z]:
                    voxel_center = [
                        min_coords[0] + x * voxel_size_x + voxel_size_x / 2,
                        min_coords[1] + y * voxel_size_y + voxel_size_y / 2,
                        min_coords[2] + z * voxel_size_z + voxel_size_z / 2
                    ]
                    centroids.append(voxel_center)
    centroids = np.array(centroids)
    return centroids

if __name__=="__main__":
    static_data_recording_path = STATIC_DATA_PATH

    parser = argparse.ArgumentParser(description='prepare data for grasp planner')

    parser.add_argument('--seed', type=int, default=2021, help='random seed for sampling')
    parser.add_argument('--num_samples', type=int, default=50, help='num grasp samples')
    parser.add_argument('--grasp_planner_data_root_path', type=str, default="/home/shinghei/Downloads/grasp_planner_data", help='path to the root folder containing grasp planner data')
    parser.add_argument('--stress_net_model_path', type=str, default="/home/shinghei/Downloads/shinghei_stuff/all_6polygon_open_gripper/epoch_193", help='path to stress net weights')
    parser.add_argument('--object_name', type=str, default="6polygon04", help='object name')
    parser.add_argument('--vis', type=int, default=0, help='0: do not visualize generated grasp; 1: visualize')
    parser.add_argument('--young_modulus', type=int, default=5, help='stiffness of the object')

    args = parser.parse_args()

    np.random.seed(args.seed)
    object_name = args.object_name
    vis = args.vis==1
    num_grasp_samples = args.num_samples
    given_young_modulus = args.young_modulus

    grasp_planner_data_root_path = args.grasp_planner_data_root_path
    grasp_data_recording_path = os.path.join(grasp_planner_data_root_path, f"{object_name}")
    os.makedirs(grasp_data_recording_path, exist_ok=True)

    model_path = args.stress_net_model_path
            
    num_pts = 1024
    

    device = torch.device("cuda")
    model = StressNet2(num_channels=5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    start_time = timeit.default_timer() 


    print("======================")

    ### Load static data
    static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, f"{object_name}.pickle"))  
    adjacent_tetrahedral_dict = static_data["adjacent_tetrahedral_dict"]
    homo_mats = static_data["homo_mats"]
    
    partial_pcs = static_data["transformed_partial_pcs"]
    pc_idx = 0
    partial_pc = partial_pcs[pc_idx:pc_idx+1,:,:]

    print(f"partial pc: {partial_pc.shape}")    

    for k in range(1):
        
        open_gripper_pc = get_gripper_point_cloud_no_transform([0.04,0.04], num_pts=num_pts)
        
        transformed_gripper_pcs = []
        gripper_pcs = []
        for i in range(1):
            transformed_gripper_pcs.append(transform_point_cloud(open_gripper_pc, homo_mats[0])[np.newaxis, :]) # gripper pc in object frame
            gripper_pcs.append(open_gripper_pc[np.newaxis, :]) # gripper pc in world frame

        transformed_open_gripper_pc = np.concatenate(tuple(transformed_gripper_pcs), axis=0)  
        open_gripper_pc =  np.concatenate(tuple(gripper_pcs), axis=0)

        augmented_transformed_gripper_pc_open = np.concatenate([transformed_open_gripper_pc, np.tile(np.array([[0, 0]]), 
                                                (1, transformed_open_gripper_pc.shape[1], 1))], axis=2)


        ########## Sample query points in a grid within the bounding box
        query = sample_grid_points_bounding_box(trimesh.PointCloud(partial_pc.squeeze()), scales=[1.2]*3, voxel_size=0.0008) # shape (num_query_pts,3)
        

        for force in [0]:   
            for young_modulus in [given_young_modulus]:  

                print(f"getting occupancy prediction for {object_name} - young {young_modulus:.3f} - force {force:.2f} started")
                                
                ### Augmented object pc
                augmented_partial_pcs = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus]]), 
                                                        (1, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
            
                ## Combine object pc and gripper pc
                combined_pcs = np.concatenate((augmented_partial_pcs, augmented_transformed_gripper_pc_open), axis=1) # shape (8, num_pts*2, 5)
                combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)

                print("combined_pc: ", combined_pc_tensor.shape)
                print("query: ", query.shape)
                
                # pass through trough the trained neural network to get occupancy as "ground truth during grasp planning"
                query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
                query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
                stress, occupancy = model(combined_pc_tensor, query_tensor)


                pred_stress = stress.squeeze().cpu().detach().numpy()
                pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
                occupied_idxs = np.where(pred_occupancy >= 0.7)[0]

        ########### sample grasp pose on object with 0 force from the predicted volumetric pc
        volumetric_pc = query[occupied_idxs]
        
        if vis:
            volumetric_pcd = pcd_ize(volumetric_pc, color=(1,0,1))
            query_pcd = pcd_ize(query, color=(0,1,0))
            object_pc_pcd = pcd_ize(partial_pc[0], color=(0,0,1))
            open3d.visualization.draw_geometries([object_pc_pcd, query_pcd])
            open3d.visualization.draw_geometries([volumetric_pcd, object_pc_pcd])
        
        ##### NOTE: THE transform IS IN WORLD FRAME and is in quaternion ####
        grasps = sample_grasps(volumetric_pc, cls_sampler="AntipodalSampler", 
                            number_of_grasps=num_grasp_samples, visualization=vis, vis_gripper_name='panda',
                            alpha=0.03)
        
        print("num grasp samples: ", len(grasps["poses"]))
        print("grasp pose length: ", len(grasps["poses"][0]))

        grasps_quaternion = deepcopy(grasps)

        for i, grasp_pose in enumerate(grasps["poses"]):
            quaternion = grasp_pose[3:]
            euler_angles = tra.euler_from_quaternion(quaternion, axes='sxyz')
            grasps["poses"][i] = [grasp_pose[0], grasp_pose[1], grasp_pose[2], euler_angles[0], euler_angles[1], euler_angles[2]]

        print("After conversion from quaternion to euler angles: ")
        print("num grasp samples: ", len(grasps["poses"]))
        print("grasp pose length: ", len(grasps["poses"][0]))

        
        # partial_pc in obj frame, gripper_pc in world frame, query in object frame
        grasp_planner_data = {"object_pc": partial_pc, "gripper_pc": open_gripper_pc, "query": query, "pred_occupancy_0": pred_occupancy, "grasp_samples": grasps, "world_to_obj_homo_mat": homo_mats[0], "volumetric_pc": volumetric_pc}

        with open(os.path.join(grasp_data_recording_path, "grasp_planner_training_data.pickle"), 'wb') as handle:
            pickle.dump(grasp_planner_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        ################# visualize #####################
        if vis:
            vis_grasp_idx = 0

            query_pcd = pcd_ize(query, color=(0,1,0))
            gripper_pc_pcd = pcd_ize(open_gripper_pc[0], color=(0,0,0))
            object_pc_pcd = pcd_ize(partial_pc[0], color=(0,0,1))

            init_grasp_pose_homo_mat = pose_to_homo_mat(torch.tensor(grasps["poses"][vis_grasp_idx][3:6]), torch.tensor(grasps["poses"][vis_grasp_idx][0:3])).numpy()
            init_grasp_pcd = transform_point_cloud(open_gripper_pc[0], init_grasp_pose_homo_mat) # transform to grasp pose
            init_grasp_pcd = transform_point_cloud(init_grasp_pcd, homo_mats[0]) # transform from world to object frame 
            init_grasp_pcd = pcd_ize(init_grasp_pcd, color=(1,0,0))

            #volumetric_pcd = pcd_ize(volumetric_pc, color=(1,0,1))

            query_occupied_pcd = pcd_ize(query[occupied_idxs], color=(1,0,1))

            open3d.visualization.draw_geometries([query_pcd, object_pc_pcd, gripper_pc_pcd, init_grasp_pcd, query_occupied_pcd])







    














