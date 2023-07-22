import torch
import os
from torch.utils.data import Dataset
import pickle   
import numpy as np  
import sys
sys.path.append("../../")
from utils.miscellaneous_utils import read_pickle_data, print_color


class StressPredictionDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        self.gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/gripper_data_sphere02"
        self.object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
        # self.gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
        # self.object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data" 
            
    def load_pickle_data(self, filename):
        # if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
        #     print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):   

        query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, f"processed sample {idx}.pickle"))  # shape (B, 3)
        object_name = query_data["object_name"]
        grasp_idx = query_data["grasp_idx"]
        force = query_data["force"]
        young_modulus = query_data["young_modulus"]
        
        # object_name = "ellipsoid03-p1"
        ### Load robot gripper point cloud
        gripper_pc = read_pickle_data(data_path=os.path.join(self.gripper_pc_path, f"{object_name}_grasp_{grasp_idx}.pickle"))["gripper_pc"]
        augmented_gripper_pc = np.hstack((gripper_pc, np.tile(np.array([0, 0]), 
                                        (gripper_pc.shape[0], 1)))) # shape (num_pts,5)
        augmented_gripper_pc = np.tile(augmented_gripper_pc[np.newaxis, :, :], (8, 1, 1)) # shape (8,num_pts,5)

        ### Load partial-view object point clouds
        if "-p1" in object_name or "-p2" in object_name:
            object_name = object_name[:-3]  # Ignore the -p1 and -p2 part.
        partial_pcs = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))["partial_pcs"]   # shape (8, num_pts, 3)
        augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)

        
        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1) # shape (B, 8, num_pts*2, 5)
        pc = torch.from_numpy(combined_pcs).permute(0,2,1).float()  # shape (B, 8, 5, num_pts*2)   
        pc = pc[0,:,:]  # just use the first point cloud                   
        

        ### Duplicate query points, stress, sign distance, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (B, num_queries, 3) FIX
        # query = query.unsqueeze(0).repeat(8,1,1)  # shape (B, 8, num_queries, 3)
        # query[..., 1] -= 1.0  # add 1.0 to each z value of each point cloud (to match with Isabella's data)
        # query[:, [1, 2]] = query[:, [2, 1]]   # swap y and z values (to match with Isabella's data) 


        signed_distance = torch.tensor(query_data["signed_distance"]).float()  # shape (B, num_queries)

        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        # occupancy = occupancy.unsqueeze(0).repeat(8,1)  # shape (B, 8, num_queries)
        

        sample = {"pc": pc, "query":  query, "occupancy": occupancy, "signed_distance": signed_distance}    

        return sample
    
