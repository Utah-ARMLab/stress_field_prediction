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
from model import StressNetOccupancyOnly



mgn_dataset_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"
raw_data_path = os.path.join(mgn_dataset_main_path, "raw_pickle_data")
filtered_data_path = os.path.join(mgn_dataset_main_path, "filtered_data")
static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data"
dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/shinghei_data"

start_time = timeit.default_timer() 
visualization = False
query_type = "sampled"  # options: sampled, full
num_pts = 1024
num_query_pts = 10000
# stress_visualization_min = 0.001   
# stress_visualization_max = 5e3 
# log_stress_visualization_min = np.log(stress_visualization_min)   
# log_stress_visualization_max = np.log(stress_visualization_max)    

grasp_idx_bounds = [0, 1]
force_levels = np.arange(1, 15.25, 0.25)  #np.arange(1, 15.25, 0.25)    [1.0]
    

device = torch.device("cuda")
model = StressNetOccupancyOnly(num_channels=5).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/new_sdf_data/run1(occ_only)/epoch 150"))
model.eval()


fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]
selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]
selected_primitive_names = ["ellipsoid"]

excluded_objects = \
[f"6polygon0{i}" for i in [1,3]] + [f"8polygon0{i}" for i in [1,3]] + \
[f"cylinder0{i}" for i in [1,2,3]] + [f"sphere0{i}" for i in [1,3]]
# print("excluded_objects:", excluded_objects)

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):
# for idx, file_name in enumerate(sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))):
for idx, file_name in enumerate(["ellipsoid01-p1"]):
    object_name = os.path.splitext(file_name)[0]

    print("======================")
    print_color(f"{object_name}, {idx}")
    print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins")

    if not any([prim_name in object_name for prim_name in selected_primitive_names]):   # if object does NOT belong to any of the selected primitives.
        print_color(f"{object_name} is not processed (type 1)")
        continue
    if any([excluded_object in object_name for excluded_object in excluded_objects]):   # if object belongs to the excluded object list.
        print_color(f"{object_name} is not processed (type 2)")
        continue 

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
        partial_pcs = read_pickle_data(data_path=os.path.join(static_data_recording_path, 
                                        f"{real_object_name}.pickle"))["partial_pcs"]   # shape (8, num_pts, 3)
        # pcds = []
        # for j, pc in enumerate(partial_pcs):
        #     pcds.append(pcd_ize(pc, color=[0,1,0]).translate((0,0.00*j,0)))
        # open3d.visualization.draw_geometries(pcds)


        for i in range(0,50):     # 50 time steps. Takes ~0.40 mins to process

            query_data = read_pickle_data(data_path=os.path.join(dataset_path, f"processed sample {i}.pickle"))  # shape (B, 3)
            test_stress = query_data["stress_log"]
            test_query = query_data["query_points"]
            num = test_stress.shape[0]
            test_occ = query_data["occupancy"]
          

            force = forces[i]  
            print("force:", force)
            full_pc = full_pcs[i]

            augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                    (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
        
            ### Combine object pc and gripper pc
            combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1)[0:1,:,:] # shape (8, num_pts*2, 5)
            combined_pc_tensor = torch.from_numpy(combined_pcs).permute(0,2,1).float().to(device)  # shape (8, 5, num_pts*2)
            
            
            ### Get query points (sample randomly or use the ground-truth particles)
            if query_type == "sampled":
                query = sample_points_bounding_box(trimesh.PointCloud(full_pc), num_query_pts, scales=[1.2]*3)  # shape (num_query_pts,3) 
                
                # query = full_pc
                # query = test_query
                is_inside = is_inside_tet_mesh_vectorized(query, vertices=full_pc, tet_indices=tet_indices).astype(int)

            pcd_test = pcd_ize(test_query[np.where(test_occ==0)[0]], color=[0,0,0], vis=True)
            
            query_tensor = torch.from_numpy(query).float()  # shape (B, num_queries, 3)
            query_tensor = query_tensor.unsqueeze(0).to(device)  # shape (8, num_queries, 3)
            occupancy = model(combined_pc_tensor, query_tensor) # shape (8*num_queries,1)

            predicted_classes = (occupancy >= 0.5).int().squeeze().cpu().detach().numpy()
            print(predicted_classes.shape, is_inside.shape)
            print("Accuracy:", np.sum(predicted_classes == is_inside)/is_inside.shape[0])

            
            # stress = stress.view(8, num_query_pts, 1)  # shape (8, num_queries, 1)
            # occupancy = stress.view(8, num_query_pts, 1)  # shape (8, num_queries, 1)
            pred_occupancy = occupancy.squeeze().cpu().detach().numpy()
            occupied_idxs = np.where(pred_occupancy >= 0.5)[0]
            # occupied_idxs = np.where(predicted_classes == is_inside)[0]
            # occupied_idxs_2 = np.where(predicted_classes == 1)[0]
            # occupied_idxs = np.intersect1d(occupied_idxs, occupied_idxs_2)
            
            # print(np.where(predicted_classes != is_inside)[0].shape, np.where(predicted_classes == is_inside)[0].shape, query[occupied_idxs].shape)       
            # print(np.count_nonzero(is_inside[occupied_idxs] == 1), np.count_nonzero(is_inside[occupied_idxs] == 0), np.count_nonzero(predicted_classes[occupied_idxs] == 1))
            
            # # print(np.sum(is_inside == test_occ), test_occ.shape[0])
            # # wrong_labels = np.where(is_inside != test_occ)[0]
            # # print(np.count_nonzero(is_inside[wrong_labels] == 1), np.count_nonzero(test_occ[wrong_labels] == 1))
            # # # print(is_inside[occupied_idxs], predicted_classes[occupied_idxs])
            # # pcd_wrong = pcd_ize(query[wrong_labels], color=[0,1,0])

            pcd_gt = pcd_ize(full_pc, color=[1,0,0])

            pcd = pcd_ize(query[occupied_idxs], color=[0,0,0])
            

            open3d.visualization.draw_geometries([pcd.translate((0.07,0,0)), pcd_gt])
            
            
            print_color("========================")
            
            

            
            
            break
        
        # break















