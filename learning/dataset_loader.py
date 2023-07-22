import torch
import os
from torch.utils.data import Dataset
import pickle   
import numpy as np  
import sys
sys.path.append("../")
from utils.miscellaneous_utils import read_pickle_data, print_color

class StressPredictionDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
            
    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):    

        sample = self.load_pickle_data(f"processed sample {idx}.pickle")

        pc = torch.from_numpy(sample["combined_pc"]).float()  # shape (B, 5, 1024+1024)               
        query = torch.from_numpy(sample["query_point"]).float()  # shape (B, 3) 
        stress = torch.FloatTensor([sample["stress_log"]])  # shape (B, 1) 
        occupancy = torch.tensor(sample["occupancy"]).unsqueeze(0).float()  # shape (B, 1)
    
        sample = {"pc": pc, "query":  query,  "stress": stress, "occupancy": occupancy}         
        return sample 


class StressPredictionDataset2(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
            
    def load_pickle_data(self, filename):
        if os.path.getsize(os.path.join(self.dataset_path, filename)) == 0: 
            print(filename)
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):    

        sample = self.load_pickle_data(f"processed sample {idx}.pickle")

        pc = torch.from_numpy(sample["combined_pc"]).float()  # shape (B, 5, 1024+1024)               
        query = torch.from_numpy(sample["query_points"]).float()  # shape (B, 3) 
        stress = torch.FloatTensor([sample["stress_log"]])  # shape (B, 1) 
        occupancy = torch.tensor(sample["occupancy"]).unsqueeze(0).float()  # shape (B, 1)
    
        sample = {"pc": pc, "query":  query,  "stress": stress, "occupancy": occupancy}         
        return sample
    
    
class StressPredictionDataset3(Dataset):

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
        pc = pc[0,:,:]  # just use the first point cloud                   
        

        ### Duplicate query points, stress, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (B, num_queries, 3) FIX
        # query[..., 1] += 0.015    # shift back to (0,0,0) origin
        # query = query.unsqueeze(0).repeat(8,1,1)  # shape (B, 8, num_queries, 3)
        
        
        # stress = torch.FloatTensor([query_data["stress_log"]])  # shape (B, 1, num_queries) FIX        
        # # # stress = stress.repeat(8,1)  # shape (B, 8, num_queries)    
        # # # stress_log = torch.log(stress)          
        # positive_mask = stress > 0.0
        # stress_log = torch.where(positive_mask, torch.log(stress), torch.log(torch.FloatTensor([0.0001])))  # Compute the logarithm of positive elements and keep negative elements unchanged. When stress = 0, just curve it up to 0.1 so log(stress) is defined
        # stress_log = stress_log.repeat(8,1)  # shape (B, 8, num_queries)


        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        # occupancy = occupancy.unsqueeze(0).repeat(8,1)  # shape (B, 8, num_queries)
        
        
        # if torch.isinf(stress_log).any():
        #     nan_indices = torch.isinf(stress_log)
        #     # stress = stress.repeat(8,1)
        #     print(torch.nonzero(torch.isinf(stress_log)).squeeze())
        #     print(stress[nan_indices])
        #     print(occupancy[nan_indices])
        #     raise SystemExit("nan")
        
        # sample = {"pc": pc, "query":  query,  "stress": stress_log, "occupancy": occupancy}     
        sample = {"pc": pc, "query":  query, "occupancy": occupancy}    

        return sample
    
class StressPredictionDataset4(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        self.gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
        self.object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data" 
            
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
        gripper_pc_tensor = torch.from_numpy(gripper_pc).permute(1,0).float()  # shape (B, 3, num_pts)
        # gripper_pc_tensor = gripper_pc_tensor.unsqueeze(0).repeat(8,1,1)  # shape (B, 8, 3, num_pts)
        

        ### Load partial-view object point clouds
        if "-p1" in object_name or "-p2" in object_name:
            object_name = object_name[:-3]  # Ignore the -p1 and -p2 part.
        partial_pcs = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))["partial_pcs"]   # shape (8, num_pts, 3)
        augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                (8, partial_pcs.shape[1], 1))], axis=2)   # shape (8, num_pts, 5)
        object_pc_tensor = torch.from_numpy(augmented_partial_pcs).permute(0,2,1).float()  # shape (B, 8, 5, num_pts)                       
        object_pc_tensor = object_pc_tensor[0,:,:]  # just use the first point cloud

        ### Duplicate query points, stress, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (B, num_queries, 3) FIX
        # query = query.unsqueeze(0).repeat(8,1,1)  # shape (B, 8, num_queries, 3)
        
        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        # occupancy = occupancy.unsqueeze(0).repeat(8,1)  # shape (B, 8, 1)
        
  
        sample = {"object_pc": object_pc_tensor, "gripper_pc": gripper_pc_tensor, "query":  query, "occupancy": occupancy}    

        return sample
    
class StressPredictionDataset5(Dataset):

    """ 
    Train occupancy only. Use force as a separate channel, NOT a feature in the point cloud anymore.
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        self.gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
        self.object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data" 
            
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


        ### Load partial-view object point clouds
        if "-p1" in object_name or "-p2" in object_name:
            object_name = object_name[:-3]  # Ignore the -p1 and -p2 part.
        partial_pcs = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))["partial_pcs"]   # shape (8, num_pts, 3)
        

        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((partial_pcs[0,:,:], gripper_pc), axis=0) # shape (B, num_pts*2, 3)
        pc = torch.from_numpy(combined_pcs).permute(1,0).float()  # shape (B, 3, num_pts*2)                          
        

        ### Duplicate query points, stress, and occupancy to accomodate 8 partial-view point clouds (from 8 different camera angles)
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (B, num_queries, 3) FIX
   
        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX

        force = torch.tensor(query_data["force"]).unsqueeze(0).float()
        
         
        sample = {"pc": pc, "query":  query, "occupancy": occupancy, "force": force}    

        return sample