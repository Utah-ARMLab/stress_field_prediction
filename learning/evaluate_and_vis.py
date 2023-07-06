import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse
import isaacgym
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES
from model import StressNet


sys.path.append("../")
from utils import *


data_recording_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/eval_data_2"
ap_save_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/attachment_points_2"
adjacent_tetrahedrals_save_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/adjacent_tetrahedrals_2"

# data_recording_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/data"
# ap_save_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/attachment_points"
# adjacent_tetrahedrals_save_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/adjacent_tetrahedrals"

start_time = timeit.default_timer() 
visualization = False
use_sample_points = True
num_pts = 1024
ROBOT_Z_OFFSET = 0.25
base_thickness = 0.005

sample_idx_bounds = [2, 4]

# Get dimensions of the objects
object_mesh_path = "/home/baothach/shape_servo_data/retraction_cutting/multi_box/evaluate/Custom_mesh_2" 
# object_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
with open(os.path.join(object_mesh_path, "primitive_dict.pickle"), 'rb') as handle:
    dimension_dict = pickle.load(handle)
    

device = torch.device("cuda")
model = StressNet(num_channels=5).to(device)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/retraction_cutting/multi_box/weights/stress_occupancy_prediction/run2/epoch 64"))
model.eval()


for i in range(*sample_idx_bounds):
# for _ in range(*sample_idx_bounds):
#     i = np.random.randint(0,9000)
    
    if i % 100 == 0:
        print(f"Sample {i} started. Time passed: {timeit.default_timer() - start_time}")
         
    file_name = os.path.join(data_recording_path, f"sample {i}.pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue   
    
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    (tet_indices, tet_stress) = data["tet"]
    tet_indices = np.array(tet_indices).reshape(-1,4)
    (tri_indices, tri_parents, tri_normals) = data["tri"]
    full_pc = data["full_pc"]
    partial_pc = data["partial_pc"]
    mp_pos = np.array(list(data["mani_point"]["pose"]["p"]))
    object_name = data["obj_name"]
    
    thickness = dimension_dict[object_name]["thickness"] 
    z_coor = base_thickness*0.5 + thickness/2*0.5
    box_shape = (dimension_dict[object_name]["height"]/2, dimension_dict[object_name]["width"]/2, dimension_dict[object_name]["thickness"]/2)

    # Get attachment points from pickle files and process
    with open(os.path.join(ap_save_path, f"{object_name}.pickle"), 'rb') as handle:
        aps = pickle.load(handle)    
    modified_aps = np.hstack((aps, np.tile(np.array([0, 0, 1]), (aps.shape[0], 1))))    # add one-hot vector [0,0,1] to each row to indicate attachment points.

    # Stress computation
    with open(os.path.join(adjacent_tetrahedrals_save_path, f"{object_name}.pickle"), 'rb') as handle:
        adjacent_tetrahedral_dict = pickle.load(handle)   
    all_stresses = compute_all_stresses(tet_stress, adjacent_tetrahedral_dict, full_pc.shape[0])
    all_stresses_log = np.log(all_stresses)

    # Down sample to 1024 points and augment pc
    partial_pc_resampled = down_sampling(partial_pc, num_pts=num_pts)
    modified_partial_pc, nearest_idxs = augment_partial_pc(partial_pc_resampled, mp_pos, num_nn=50)
   
    
   
    if use_sample_points:
        sampled_points, outside_mesh_idxs, inside_mesh_idxs = sample_and_compute_signed_distance(tri_indices, full_pc, \
                                        boundary_threshold=[0.02,-0.01], obj_position=(0.0,-0.42,z_coor), box_shape=box_shape, \
                                        num_pts=10000, scales=(2, 2, 4), vis=False, seed=None, verbose=True)
        predicted_stresses_log, predicted_occupancies, occupied_idxs = \
                                get_stress_occupancy_query_points(model, device, query_list=sampled_points, augmented_partial_pc=modified_partial_pc, \
                                augmented_aps=modified_aps, batch_size=300, occupancy_threshold=0.99)
        
        selected_points = sampled_points[occupied_idxs] # select points that belong to the volume of the object (occupied)
        selected_stresses_log = predicted_stresses_log[occupied_idxs]
        print("len(selected_points):", len(selected_points))
        
        pcd_predicted = pcd_ize(selected_points)
        predicted_colors = np.array(scalar_to_rgb(selected_stresses_log, colormap='jet', min_val=min(all_stresses_log), max_val=max(all_stresses_log)))[:,:3]
        # predicted_colors = np.array(scalar_to_rgb(selected_stresses_log, colormap='jet'))[:,:3]
        pcd_predicted.colors = open3d.utility.Vector3dVector(predicted_colors)

        down_sampled_partial_pc = down_sampling(partial_pc, num_pts)
        predicted_stresses_log, predicted_occupancies, occupied_idxs = \
                                get_stress_occupancy_query_points(model, device, query_list=down_sampled_partial_pc, augmented_partial_pc=modified_partial_pc, \
                                augmented_aps=modified_aps, batch_size=300, occupancy_threshold=0.99)

        pcd_predicted_partial = pcd_ize(down_sampled_partial_pc)
        predicted_partial_colors = np.array(scalar_to_rgb(predicted_stresses_log, colormap='jet', min_val=min(all_stresses_log), max_val=max(all_stresses_log)))[:,:3]
        # predicted_partial_colors = np.array(scalar_to_rgb(predicted_stresses_log, colormap='jet'))[:,:3]
        pcd_predicted_partial.colors = open3d.utility.Vector3dVector(predicted_partial_colors)

    else:
        predicted_stresses_log = get_stress_query_points(model, device, query_list=full_pc, augmented_partial_pc=modified_partial_pc, \
                                augmented_aps=modified_aps, batch_size=256)        
        pcd_predicted = pcd_ize(full_pc)
        predicted_colors = np.array(scalar_to_rgb(predicted_stresses_log, colormap='jet'))[:,:3]
        pcd_predicted.colors = open3d.utility.Vector3dVector(predicted_colors)    

    pcd = pcd_ize(full_pc)
    colors = np.array(scalar_to_rgb(all_stresses_log, colormap='jet'))[:,:3]
    pcd.colors = open3d.utility.Vector3dVector(colors)
    

    
    open3d.visualization.draw_geometries([deepcopy(pcd_predicted).translate((0.3,0,0)), pcd, pcd_predicted_partial.translate((-0.3,0,0))])


 
        





    















