import os
import pickle5 as pickle
import matplotlib.pyplot as plt
from grasp_planner import *
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize, print_color
from utils.mesh_utils import open3d_to_trimesh_mesh
from utils.hands import create_gripper
import open3d
import argparse
import math

sys.path.append("../")
sys.path.append("../graspsampling-py-defgraspsim")
from graspsampling import collision
import trimesh.transformations as tra

'''
Visualize the best optimized grasp pose
'''

def get_object_mesh(grasp_planner_training_data_path):
    with open(grasp_planner_training_data_path, 'rb') as handle:
        data = pickle.load(handle) 
    volumetric_pc = data["volumetric_pc"]
    alpha = 0.03
    reconstructed_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ize(volumetric_pc), 
                                                                            alpha)
    object_mesh = open3d_to_trimesh_mesh(reconstructed_mesh)
    return object_mesh

def is_grasp_collision_free(gripper_euler_angles, gripper_translation_vec, object_mesh):
    gripper = create_gripper('panda', [0.04,0.04], 
                             franka_gripper_mesh_main_path="../graspsampling-py-defgraspsim", finger_only=True)
    roll, pitch, yaw = gripper_euler_angles.numpy()
    x, y, z = gripper_translation_vec.numpy()
    quaternion = tra.quaternion_from_euler(roll, pitch, yaw)
    gripper_poses_quaternion = {"poses": [[x, y, z, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]]}
    in_collision = collision.in_collision_with_gripper(
                gripper, object_mesh, **gripper_poses_quaternion
            )
    closing_region_nonempty = collision.check_gripper_nonempty(
        gripper, object_mesh, **gripper_poses_quaternion
    )

    mask = closing_region_nonempty & ~in_collision
    return mask[0], gripper_poses_quaternion




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='training for grasp planner')
    parser.add_argument('--num_samples', type=int, default=50, help='number of grasp samples')
    parser.add_argument('--is_max_deformation', type=int, default=0, help='0: minimize deformation, 1: maximize deformation')
    parser.add_argument('--object_name', type=str, default="6polygon04", help='object name')
    parser.add_argument('--grasp_planner_data_root_path', type=str, default="/home/shinghei/Downloads/grasp_planner_data", help='path to the root folder containing grasp planner training data')
    parser.add_argument('--stress_net_model_path', type=str, default="/home/shinghei/Downloads/shinghei_stuff(1)/all_6polygon_open_gripper/epoch 193", help='path to stress net weights')
    parser.add_argument('--last_epoch', type=int, default=150, help='last epoch of the training which we want to visualize')
    parser.add_argument('--force', type=int, default=15, help='force exerted on the object')
    parser.add_argument('--young_modulus', type=int, default=5, help='stiffness of the object')

    args = parser.parse_args()

    num_samples = args.num_samples
    maximize_deformation = args.is_max_deformation==1
    epoch = args.last_epoch

    object_name = args.object_name
    grasp_planner_data_root_path = args.grasp_planner_data_root_path
    grasp_planner_training_data_path = os.path.join(grasp_planner_data_root_path, f"{object_name}/grasp_planner_training_data.pickle")
    
    model_path = args.stress_net_model_path
    
    if maximize_deformation:
        optimized_data_suffix = "max_deformation"
    else:
        optimized_data_suffix = "min_deformation"

    force = args.force
    young_modulus = args.young_modulus
    ### B should always be 1
    B=1
    

    ###get the object mesh for collision checking 
    object_mesh = get_object_mesh(grasp_planner_training_data_path)

    ########################### choose the best grasp ####################################
    min_loss = math.inf
    best_init_grasp_idx = 0
    count_valid = 0
    for init_grasp_idx in range(num_samples):
        grasp_planner_optimized_pose_path = os.path.join(grasp_planner_data_root_path, object_name, f"optimized_pose_init_{init_grasp_idx}_{optimized_data_suffix}") 
        with open(os.path.join(grasp_planner_optimized_pose_path, f"epoch {epoch}.pickle"), 'rb') as handle:
            optimize_data = pickle.load(handle)
        
        optimized_euler_angles = optimize_data["optimized_euler_angles"].to("cpu")
        optimized_translation_vec = optimize_data["optimized_translation_vec"].to("cpu")

        is_valid_grasp, gripper_poses_quaternion = is_grasp_collision_free(optimized_euler_angles, optimized_translation_vec, object_mesh)
        count_valid += int(is_valid_grasp)
        print(f"--------------------------grasp idx: {init_grasp_idx}----------------------------------")
        print(f"is valid grasp?: {is_valid_grasp}")

        if not is_valid_grasp:
            continue
 
        if optimize_data["loss"] <= min_loss:
            best_init_grasp_idx = init_grasp_idx
            min_loss = optimize_data["loss"]

    print(f"++++++ object {object_name}: Best init grasp idx: {best_init_grasp_idx} with loss {min_loss} ********")
    print("total valid grasp count: ", count_valid)

   
    ############### load and prepare data ###################

    device = torch.device("cuda")
    best_grasp_planner_optimized_pose_path = os.path.join(grasp_planner_data_root_path, object_name, f"optimized_pose_init_{best_init_grasp_idx}_{optimized_data_suffix}") 

    with open(os.path.join(best_grasp_planner_optimized_pose_path, f"epoch {epoch}.pickle"), 'rb') as handle:
        optimize_data = pickle.load(handle) 


    #######################################################

    with open(grasp_planner_training_data_path, 'rb') as handle:
        data = pickle.load(handle) 

    partial_pc = data["object_pc"] # shape (1, num_pts, 3)
    gripper_pc = data["gripper_pc"] # shape (1, num_pts, 3)
    query = data["query"] # shape (num_queries, 3)
    occupancy_0_force = data["pred_occupancy_0"] # shape (num_queries,)
    grasp_samples = data["grasp_samples"]["poses"] #shape (num_samples, 6)
    world_to_obj_homo_mat = data["world_to_obj_homo_mat"]

    augmented_partial_pc = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus]]), 
                                                        (1, partial_pc.shape[1], 1))], axis=2) 
    
    augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
                                                (1, gripper_pc.shape[1], 1))], axis=2)
    
    augmented_partial_pc = torch.from_numpy(augmented_partial_pc).float().to(device).squeeze(0)
    augmented_gripper_pc = torch.from_numpy(augmented_gripper_pc).float().to(device).squeeze(0)
    query = torch.from_numpy(query).float().to(device).unsqueeze(0).expand(B, -1, -1)
    occupancy_0_force = torch.from_numpy(occupancy_0_force).float().to(device).unsqueeze(0).expand(B, -1)
    world_to_obj_homo_mat = torch.from_numpy(world_to_obj_homo_mat).float().to(device)

    model = StressNet2(num_channels=5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    

    ##################### visualize init gripper pose (red), learned gripper pose (black) and object partial pc (green) ###################################
    optimized_euler_angles = optimize_data["optimized_euler_angles"].to(device)
    optimized_translation_vec = optimize_data["optimized_translation_vec"].to(device)

    optimized_homo_mat = pose_to_homo_mat(optimized_euler_angles, translation_vec=optimized_translation_vec)
    optimized_transformed_gripper_pc = transform_point_torch(augmented_gripper_pc[:, :3], optimized_homo_mat)
    optimized_transformed_gripper_pc = transform_point_torch(optimized_transformed_gripper_pc[:, :3], world_to_obj_homo_mat)
    optimized_pcd_gripper = pcd_ize(optimized_transformed_gripper_pc.cpu().numpy(), color=(0,0,0))

    partial_pc = augmented_partial_pc[:, :3]
    pcd_object = pcd_ize(partial_pc.cpu().numpy(), color=(0,1,0))

    init_euler_angles = torch.tensor(grasp_samples[init_grasp_idx][3:]).float().to(device)
    init_translation_vec = torch.tensor(grasp_samples[init_grasp_idx][0:3]).float().to(device)

    init_homo_mat = pose_to_homo_mat(init_euler_angles, translation_vec=init_translation_vec)
    init_transformed_gripper_pc = transform_point_torch(augmented_gripper_pc[:, :3], init_homo_mat)
    init_transformed_gripper_pc = transform_point_torch(init_transformed_gripper_pc[:, :3], world_to_obj_homo_mat)
    init_pcd_gripper = pcd_ize(init_transformed_gripper_pc.cpu().numpy(), color=(1,0,0))

    coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    query_squeezed = query.squeeze(0)


    print(" ####### visualize initial grasp pose (red) and optimized grasp pose (black): ")
    # grasp_planner = GraspPlanner(init_euler_angles.cpu().numpy(), init_translation_vec.cpu().numpy()).to(device)
    # predicted_occ = grasp_planner(query, augmented_partial_pc, augmented_gripper_pc, model, world_to_obj_homo_mat).squeeze(0)
    # occupied_idxs = torch.where(predicted_occ >= 0.7)[0]
    # occupied_query = query_squeezed[occupied_idxs]

    # occupied_query_pcd = pcd_ize(occupied_query.cpu().numpy(), color=(1, 0, 1))
    # open3d.visualization.draw_geometries([pcd_object, init_pcd_gripper, occupied_query_pcd, coor])
    open3d.visualization.draw_geometries([pcd_object, init_pcd_gripper, optimized_pcd_gripper,coor])


    #print(" ####### visualize optimized grasp pose (black): ")
    # grasp_planner = GraspPlanner(learned_euler_angles.cpu().numpy(), learned_translation_vec.cpu().numpy()).to(device)
    # predicted_occ = grasp_planner(query, augmented_partial_pc, augmented_gripper_pc, model, world_to_obj_homo_mat).squeeze(0)
    # occupied_idxs = torch.where(predicted_occ >= 0.7)[0]
    # occupied_query = query_squeezed[occupied_idxs]

    # occupied_query_pcd = pcd_ize(occupied_query.cpu().numpy(), color=(1, 0, 1))
    # open3d.visualization.draw_geometries([pcd_object, learned_pcd_gripper, occupied_query_pcd, coor])
    #open3d.visualization.draw_geometries([pcd_object, optimized_pcd_gripper, coor])



   
                            
           


