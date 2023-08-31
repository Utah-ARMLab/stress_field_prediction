import torch
import os
from torch.utils.data import Dataset
import pickle5 as pickle   
import numpy as np  
import sys
sys.path.append("../")
from utils.miscellaneous_utils import read_pickle_data, print_color, pcd_ize
from utils.point_cloud_utils import transform_point_cloud, transform_point_clouds
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
        
        num_partial_pc = 8  # 8 1

        query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, self.file_names[idx]))  # shape (B, 3)
        object_name = query_data["object_name"]
        grasp_idx = query_data["grasp_idx"]
        force = query_data["force"]
        # young_modulus = query_data["young_modulus"]
        young_modulus = query_data["young_modulus"]/1e4
        # print("young_modulus", young_modulus)
        
        ### Load robot gripper point cloud
        gripper_pcs = read_pickle_data(data_path=os.path.join(self.gripper_pc_path, f"open_gripper_data_{object_name}", 
                                    f"{object_name}_grasp_{grasp_idx}.pickle"))["transformed_gripper_pcs"][0:num_partial_pc]   # shape (8, num_pts, 3)
        # pcd_ize(gripper_pcs[0], color=[0,0,0], vis=True)
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


class StressPredictionObjectFrameDataset_2(Dataset):

    """
    Use 100 partial point clouds, instead of just 8 like before
    Dataset for training either only occupancy, or stress and occupancy jointly. Using 8 partial pc from 8 camera views, transformed to object frame.
    """

    def __init__(self, dataset_path, gripper_pc_path, object_partial_pc_path, object_names, 
                 num_partial_pc = 1,
                 joint_training=True):
        self.dataset_path = dataset_path
        self.gripper_pc_path = gripper_pc_path
        self.object_partial_pc_path = object_partial_pc_path   
        self.object_names = object_names

        self.num_partial_pc = num_partial_pc
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

        ### Query points, stress, and occupancy
        query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, self.file_names[idx]))  # shape (B, 3)
        object_name = query_data["object_name"]
        grasp_idx = query_data["grasp_idx"]
        force = query_data["force"]
        young_modulus = query_data["young_modulus"]/1e4
        

        ### Load partial-view object point clouds   
        static_data = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))
        homo_mats = static_data["homo_mats"]   # shape (num_pcs, 4, 4)
        partial_pcs = static_data["transformed_partial_pcs"]   # shape (num_pcs, num_pts, 3)
        # print_color(f"Total num partial pcs: {partial_pcs.shape[0]}")
        
        selected_partial_pc_idxs = np.random.randint(low=0, high=partial_pcs.shape[0], size=self.num_partial_pc)
        # print(f"selected idx: {selected_partial_pc_idxs}")
        partial_pcs = partial_pcs[selected_partial_pc_idxs,:,:]
        homo_mats = homo_mats[selected_partial_pc_idxs,:,:]
        
        augmented_partial_pcs = np.concatenate([partial_pcs, np.tile(np.array([[force, young_modulus]]), 
                                                (self.num_partial_pc, partial_pcs.shape[1], 1))], axis=2)   # shape (num_pcs, num_pts, 5)

        ### Load robot gripper point cloud and transform
        gripper_pc = read_pickle_data(data_path=os.path.join(self.gripper_pc_path, f"open_gripper_data_{object_name}", 
                                    f"{object_name}_grasp_{grasp_idx}.pickle"))["gripper_pc"]  # shape (num_pts, 3)
        transformed_gripper_pcs = transform_point_clouds(gripper_pc, homo_mats)
        
        augmented_gripper_pcs = np.concatenate([transformed_gripper_pcs, np.tile(np.array([[0, 0]]), 
                                                (self.num_partial_pc, transformed_gripper_pcs.shape[1], 1))], axis=2)   # shape (num_pcs, num_pts, 5)       

        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((augmented_partial_pcs, augmented_gripper_pcs), axis=1) # shape (B, num_pcs, num_pts*2, 5)
        pc = torch.from_numpy(combined_pcs).permute(0,2,1).float()  # shape (B, num_pcs, 5, num_pts*2)   
      
    
        ### Duplicate query points, stress, and occupancy to accomodate num_pcs partial-view point clouds (from num_pcs different camera angles)    
        transformed_qrs = transform_point_clouds(query_data["query_points"], homo_mats)  # shape (num_pcs, num_pts, 3)  
        query = torch.from_numpy(transformed_qrs).float()  # shape (B, num_pcs, num_queries, 3)


        if self.joint_training:
            stress_log = torch.FloatTensor([query_data["stress_log"]])  # shape (B, 1, num_queries) FIX        
            stress_log = stress_log.repeat(self.num_partial_pc,1)  # shape (B, num_pcs, num_queries)

        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (B, num_queries) FIX
        occupancy = occupancy.unsqueeze(0).repeat(self.num_partial_pc,1)  # shape (B, num_pcs, num_queries)
        
        if self.joint_training:
            sample = {"pc": pc, "query":  query,  "stress": stress_log, "occupancy": occupancy}   
        else:
            sample = {"pc": pc, "query":  query, "occupancy": occupancy}    

        return sample   
    

class StressPredictionFullMesh(Dataset):

    """
    Dataset for full mesh training. 
    """

    def __init__(self, dataset_path, gripper_pc_path, object_partial_pc_path, object_names, 
                 joint_training=True):
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

        ### Query points, stress, and occupancy
        query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, self.file_names[idx]))  # shape (B, 3)
        object_name = query_data["object_name"]
        grasp_idx = query_data["grasp_idx"]
        force = query_data["force"]
        young_modulus = query_data["young_modulus"]/1e4
        

        ### Load partial-view object point clouds   
        static_data = read_pickle_data(data_path=os.path.join(self.object_partial_pc_path, 
                                        f"{object_name}.pickle"))
        # mesh = static_data["downsampled_mesh_vertices"]   # shape (num_pts, 3)
        mesh = static_data["partial_pcs"][0]   # shape (num_pts, 3)
        
        augmented_mesh = np.concatenate([mesh, np.tile(np.array([[force, young_modulus]]), 
                                        (mesh.shape[0], 1))], axis=1)   # shape (num_pcs, num_pts, 5)


        ### Load robot gripper point cloud and transform    open_gripper_data_{object_name}
        gripper_pc = read_pickle_data(data_path=os.path.join(self.gripper_pc_path, f"gripper_data_{object_name}", 
                                    f"{object_name}_grasp_{grasp_idx}.pickle"))["gripper_pc"]  # shape (num_pts, 3)      
        augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
                                            (gripper_pc.shape[0], 1))], axis=1)   # shape (num_pts, 5)       
        # import open3d
        # mesh_pcd = pcd_ize(mesh, color=[0,0,0], vis=False)
        # query_data = read_pickle_data(data_path=os.path.join(self.dataset_path, f"processed_data_{object_name}/processed sample 0.pickle"))
        # query_pcd = pcd_ize(query_data["query_points"][:2000], color=[1,0,0], vis=False)
        # gripper_pcd = pcd_ize(gripper_pc, color=[0,0,1], vis=False)
        # open3d.visualization.draw_geometries([mesh_pcd, query_pcd, gripper_pcd])

        ### Combine object pc and gripper pc
        combined_pcs = np.concatenate((augmented_mesh, augmented_gripper_pc), axis=0) 
        pc = torch.from_numpy(combined_pcs).permute(1,0).float()  # shape (5, num_pts*2)   
      
    
        ### Process query points, stress, and occupancy
        query = torch.from_numpy(query_data["query_points"]).float()  # shape (num_queries, 3)


        if self.joint_training:
            stress_log = torch.FloatTensor([query_data["stress_log"]])  # shape (num_queries,)      

        occupancy = torch.tensor(query_data["occupancy"]).float()  # shape (num_queries,) 
        
        if self.joint_training:
            sample = {"pc": pc, "query":  query,  "stress": stress_log, "occupancy": occupancy}   
        else:
            sample = {"pc": pc, "query":  query, "occupancy": occupancy}    

        return sample 