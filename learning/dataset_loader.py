import torch
import os
from torch.utils.data import Dataset
import pickle   
import numpy as np  
import sys
sys.path.append("../")
from utils.miscellaneous_utils import read_pickle_data, print_color
from utils.point_cloud_utils import transform_point_cloud
import random

   
    
class StressPredictionDataset3(Dataset):

    """
    Dataset for training only occupancy. Using 8 partial pc from 8 camera views, using world frame.
    """

    def __init__(self, dataset_path, gripper_pc_path, object_partial_pc_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        # self.gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
        # self.object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data" 
        self.gripper_pc_path = gripper_pc_path
        self.object_partial_pc_path = object_partial_pc_path        


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
        # partial_pcs[..., 1] -= 1.0
        augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)

        
        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pc), axis=1) # shape (B, 8, num_pts*2, 5)
        pc = torch.from_numpy(combined_pcs).permute(0,2,1).float()  # shape (B, 8, 5, num_pts*2)   
        # pc = pc[0,:,:]  # just use the first point cloud                   
        

        ### Duplicate query points, stress, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (B, num_queries, 3) FIX
        # query[..., 1] += 1.0    # shift back to (0,0,0) origin
        query = query.unsqueeze(0).repeat(8,1,1)  # shape (B, 8, num_queries, 3)
        
        
        stress_log = torch.FloatTensor([query_data["stress_log"]])  # shape (B, 1, num_queries) FIX        
        stress_log = stress_log.repeat(8,1)  # shape (B, 8, num_queries)


        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        occupancy = occupancy.unsqueeze(0).repeat(8,1)  # shape (B, 8, num_queries)
        
        
       
        sample = {"pc": pc, "query":  query,  "stress": stress_log, "occupancy": occupancy}     
        # sample = {"pc": pc, "query":  query, "occupancy": occupancy}    

        return sample
    

class StressPredictionObjectFrameDataset(Dataset):

    """
    Dataset for training either only occupancy, or stress and occupancy jointly. Using 8 partial pc from 8 camera views, transformed to object frame.
    """

    def __init__(self, dataset_path, gripper_pc_path, object_partial_pc_path, object_names, joint_training=True):
        self.dataset_path = dataset_path
        self.gripper_pc_path = gripper_pc_path
        self.object_partial_pc_path = object_partial_pc_path   
        self.object_names = object_names
        self.joint_training = joint_training    # bool: training either only occupancy, or stress and occupancy jointly     

        self.file_names = []
        for object_name in object_names:            
            self.file_names += [os.path.join(f"processed_data_{object_name}", file) for file 
                               in os.listdir(os.path.join(self.dataset_path, f"processed_data_{object_name}"))]
    
        random.shuffle(self.file_names)

    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):   
        
        num_partial_pc = 1  # 8

        query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, self.file_names[idx]))  # shape (B, 3)
        object_name = query_data["object_name"]
        grasp_idx = query_data["grasp_idx"]
        force = query_data["force"]
        # young_modulus = query_data["young_modulus"]
        young_modulus = np.exp(query_data["young_modulus"])/1e4
        # print("young_modulus", young_modulus)
        
        ### Load robot gripper point cloud
        gripper_pcs = read_pickle_data(data_path=os.path.join(self.gripper_pc_path, f"gripper_data_{object_name}", 
                                    f"{object_name}_grasp_{grasp_idx}.pickle"))["transformed_gripper_pcs"][0:num_partial_pc]   # shape (8, num_pts, 3)
        augmented_gripper_pcs = np.concatenate([gripper_pcs, np.tile(np.array([[0, 0]]), 
                                                (num_partial_pc, gripper_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)        
        

        ### Load partial-view object point clouds   
        if "-p1" in object_name or "-p2" in object_name:
            object_name = object_name[:-3]  # Ignore the -p1 and -p2 part.
        static_data = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))
        homo_mats = static_data["homo_mats"]    # list of 8 4x4 homo transformation matrices
        partial_pcs = static_data["transformed_partial_pcs"][0:num_partial_pc]   # shape (8, num_pts, 3)
        augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                (num_partial_pc, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)

        
        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pcs), axis=1) # shape (B, 8, num_pts*2, 5)
        pc = torch.from_numpy(combined_pcs).permute(0,2,1).float()  # shape (B, 8, 5, num_pts*2)   
        # pc = pc[0,:,:]  # just use the first point cloud                   
        

        ### Duplicate query points, stress, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        transformed_qrs = []
        for i in range(num_partial_pc):
            transformed_qrs.append(transform_point_cloud(query_data["query_points"], homo_mats[i])[np.newaxis, :])     
        transformed_qrs = np.concatenate(tuple(transformed_qrs), axis=0)  
        query = torch.from_numpy(transformed_qrs).float()  # shape (B, 8, num_queries, 3)


        if self.joint_training:
            stress_log = torch.FloatTensor([query_data["stress_log"]])  # shape (B, 1, num_queries) FIX        
            stress_log = stress_log.repeat(num_partial_pc,1)  # shape (B, 8, num_queries)

        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        occupancy = occupancy.unsqueeze(0).repeat(num_partial_pc,1)  # shape (B, 8, num_queries)
        
        if self.joint_training:
            sample = {"pc": pc, "query":  query,  "stress": stress_log, "occupancy": occupancy}   
        else:
            sample = {"pc": pc, "query":  query, "occupancy": occupancy}    

        return sample   


