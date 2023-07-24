import open3d
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
from model import StressNetOccupancyOnly3
from copy import deepcopy



mgn_dataset_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"
raw_data_path = os.path.join(mgn_dataset_main_path, "raw_pickle_data")
filtered_data_path = os.path.join(mgn_dataset_main_path, "filtered_data")
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"
dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/ellipsoid01_test_all"

start_time = timeit.default_timer() 
visualization = False
query_type = "sampled"  # options: sampled, full
num_pts = 1024
num_query_pts = 10000

    

device = torch.device("cuda")
model = StressNetOccupancyOnly3(num_channels=5).to(device)   # run5(occ_only_p1) run4(occ_only_6polygon04)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/new/sphere02/epoch 600"))
model.eval()





for idx, file_name in enumerate(["sphere02"]):    #"ellipsoid01-p1" "6polygon04"
    object_name = os.path.splitext(file_name)[0]

    print("======================")
    print_color(f"{object_name}, {idx}")
    print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")


    for k in range(0,100):    # 100 grasp poses
        
        file_name = os.path.join(filtered_data_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            
            print_color(f"{file_name} not found")
            continue
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
  
        ### Get data for all 50 time steps
        full_pcs = data["object_particle_states"] 
        gt_stresses = np.log(data["stresses"])
        forces = data["forces"]          
        tet_indices = data["tet_indices"]
        young_modulus = data["young_modulus"]
        # print(forces)
        
        
        ### Load robot gripper point cloud
        gripper_pc = data["gripper_pc"]     
        augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), 
                                        (gripper_pc.shape[0], 1)))) # shape (num_pts,5)
        augmented_gripper_pc = np.tile(augmented_gripper_pc[np.newaxis, :, :], (8, 1, 1)) # shape (8,num_pts,5)
        pcd_gripper = pcd_ize(gripper_pc, color=(0,0,0))



        ### Load partial-view object point clouds
        real_object_name = object_name
        if "-p1" in object_name or "-p2" in object_name:
            real_object_name = real_object_name[:-3]  # Ignore the -p1 and -p2 part.
        static_data = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{real_object_name}.pickle"))   # shape (8, num_pts, 3)
        partial_pcs = static_data["partial_pcs"]
        # partial_pcs[..., 1] -= 1.0
        tri_indices = static_data["tri_indices"]
        # pcds = []
        # for j, pc in enumerate(partial_pcs):
        #     pcds.append(pcd_ize(pc, color=[0,1,0]).translate((0,0.00*j,0)))
        # open3d.visualization.draw_geometries(pcds)

        selected_idxs = [0,10,14,15,17,18,20,21,22,23,26,27,29,30,32,33,35,36,38,40,41,43,44,46,47,49]
        pcds = []
        pcd_gts = []

        # for i in [0,len(selected_idxs)-1]:
        for i in [0,25]:
        # for i in range(0,len(selected_idxs)): #len(selected_idxs)
            
        # for i in range(49,50):     # 50 time steps. Takes ~0.40 mins to process

            i = selected_idxs[i]

            # query_data = read_pickle_data(data_path=os.path.join(dataset_path, f"processed sample {i}.pickle"))  # shape (B, 3)
            # # test_stress = query_data["stress_log"]
            # test_query = query_data["query_points"]
            # # num = test_stress.shape[0]
            # test_occ = query_data["occupancy"]
          
            

            force = forces[i]  
            print("force:", force)
            
            force_tensor = torch.tensor([force]).unsqueeze(0).float().to(device)
            
            full_pc = full_pcs[i]

        
            ### Combine object pc and gripper pc
            combined_pcs = np.concatenate((partial_pcs[0,:,:], gripper_pc), axis=0) # shape (1, num_pts*2, 3)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(1,0).unsqueeze(0).float().to(device)  # shape (1, 3, num_pts*2) 
            
            
            ### Get query points (sample randomly or use the ground-truth particles)
            if query_type == "sampled":
                query = sample_points_bounding_box(trimesh.PointCloud(full_pcs[0]), num_query_pts, scales=[1.2]*3) - np.array([0,1.0,0]) # shape (num_query_pts,3)  
                # query = full_pc - np.array([0,1.0,0])
                # query = np.concatenate((full_pc, sample_points_bounding_box(trimesh.PointCloud(full_pc), num_query_pts, scales=[1.5]*3)), axis=0)  - np.array([0,1.0,0])

                # query[..., 1] -= 1.0
                # query[..., 1] += 0.015
                
                # object_mesh = trimesh.Trimesh(vertices=full_pc, faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))
                # signed_distance_full_pc = trimesh.proximity.signed_distance(object_mesh, full_pc)
                # print("min sd:", min(signed_distance_full_pc))
                # query, signed_distances_random, \
                # outside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                #                     boundary_threshold=[0.02, min(signed_distance_full_pc)], \
                #                     num_pts=round(num_query_pts), scales=[1.5]*3, vis=False, seed=None, verbose=False)      

                
                
                # num_test_query = test_query.shape[0]
                # query = test_query
                # query = test_query[:round(num_test_query/2)]
                # # # is_inside = test_occ
                
                is_inside = is_inside_tet_mesh_vectorized(query, vertices=full_pc, tet_indices=tet_indices).astype(int)
                # signed_distance_query = trimesh.proximity.signed_distance(object_mesh, query)
                # is_inside = signed_distances_random >= 0.0
                # count_inside = np.where(is_inside)[0].shape[0]
                # print("test", np.where(test_occ[:4000])[0].shape[0])
                
                # is_inside = test_occ

            # print("count_inside:", count_inside)
            # print(trimesh.proximity.signed_distance(object_mesh, query[np.where(is_inside == False)[0]]))
            # pcd_test = pcd_ize(query[np.where(is_inside == True)[0]], color=[0,0,0], vis=True)
            # pcd_test = pcd_ize(test_query[np.where(test_occ==1)[0]], color=[0,0,0], vis=False)
            
            query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
            query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
            print(combined_pc_tensor.shape, force_tensor.shape)
            occupancy = model(combined_pc_tensor, force_tensor, query_tensor) # shape (8*num_queries,1)

            predicted_classes = (occupancy >= 0.5).int().squeeze().cpu().detach().numpy()
            # print(predicted_classes.shape, is_inside.shape)m
            # print("Accuracy:", np.sum(predicted_classes == is_inside)/is_inside.shape[0])


            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.5)[0]
            print("occupied_idxs.shape[0] / total:", occupied_idxs.shape[0], query.shape[0])


            # top_indices = np.argsort(pred_occupancy[occupied_idxs])[round(occupied_idxs.shape[0]*0.0):]
            # pcd_top = pcd_ize(query[occupied_idxs[top_indices]], color=[0,0,1], vis=False)

            # bad_idxs = np.where(is_inside != (pred_occupancy >= 0.5))[0]
            # pcd_bad = pcd_ize(query[bad_idxs], color=[0,1,0], vis=False)
            # pcd_query = pcd_ize(query, color=[0,0,0], vis=False)
            # open3d.visualization.draw_geometries([pcd_gripper, pcd_query, pcd_bad])

            # pcd_pos_query = pcd_ize(test_query[test_occ==1], color=[0,0,0])

            pcd_partial = pcd_ize(partial_pcs[3:4].reshape(-1,3), color=[0,1,0]) 

            pcd_gt = pcd_ize(full_pc - np.array([0,1.0,0]), color=[1,0,0])
            pcd_gts.append(deepcopy(pcd_gt).translate((0.07,0,0)))

            predicted_volume = query[occupied_idxs] 
            pcd = pcd_ize(predicted_volume - np.array([0,0.00,0]), color=[0,0,1])
            
            # pcd_partial = pcd_ize(partial_pcs[0], color=[0,1,0])
            # open3d.visualization.draw_geometries([pcd_gt])
            # open3d.visualization.draw_geometries([pcd.translate((0.09,0,0)), pcd_partial, pcd_gt, pcd_pos_query])
            # open3d.visualization.draw_geometries([deepcopy(pcd).translate((0.07,0,0)), pcd_gt])
            # open3d.visualization.draw_geometries([deepcopy(pcd_top).translate((0.07,0,0)), pcd_gt])


            pcds.append(pcd)
            
            # print("min y max y predicted:", min(predicted_volume[:,1]), max(predicted_volume[:,1]))
            # print("min y max y gt:", min((full_pc - np.array([0,1.0,0]))[:,1]), max((full_pc - np.array([0,1.0,0]))[:,1]))

            print_color("========================")
            
            

            
            
            # break
        
        # pcds[0].paint_uniform_color([1,0,0])
        # pcds[1].paint_uniform_color([0,0,1])
        # pcd_gts[0].paint_uniform_color([1,0,0])
        # pcd_gts[1].paint_uniform_color([0,0,1])
        # # pcds[1].translate((-0.07,0,0))
        # # pcd_gts[1].translate((0.07,0,0))
        # open3d.visualization.draw_geometries(pcds + pcd_gts)

        pcds[0].paint_uniform_color([1,0,0])
        pcds[1].paint_uniform_color([0,0,1])
        for i, pcd in enumerate(pcds):
            pcd.translate((0.0,0.00*(i),0))

        pcd_final = pcd_ize(full_pcs[-1] - np.array([0,1.0,0]), color=[1,0,0])
        pcd_final.translate((0.07,0.00,0))
        pcds.append(pcd_final)

        open3d.visualization.draw_geometries(pcds)

        break















