import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import StressNet2
import numpy as np
import timeit

'''
grasp planner that does gradient descent on the grasp pose to maximize/minimize deformation
'''

def euler_to_rotation_matrix(euler_angles):
    roll, pitch, yaw = euler_angles

    # Create individual rotation matrices for each axis
    rotation_x = torch.tensor([[1, 0, 0],
                               [0, torch.cos(roll), -torch.sin(roll)],
                               [0, torch.sin(roll), torch.cos(roll)]]).to(euler_angles.device)

    rotation_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                               [0, 1, 0],
                               [-torch.sin(pitch), 0, torch.cos(pitch)]]).to(euler_angles.device)

    rotation_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                               [torch.sin(yaw), torch.cos(yaw), 0],
                               [0, 0, 1]]).to(euler_angles.device)
    

    # Combine the rotation matrices to get the final rotation matrix
    rotation_matrix = torch.mm(rotation_z, torch.mm(rotation_y, rotation_x))

    return rotation_matrix




def pose_to_homo_mat(euler_angles: torch.Tensor, translation_vec: torch.Tensor):
    '''
    euler angles: [roll, pitch, yaw]
    translatioin_vec: [x, y, z]
    '''
    rotation_mat = euler_to_rotation_matrix(euler_angles) # shape (3, 3)
    translation_vec = translation_vec.unsqueeze(1)
    homo_mat = torch.cat((rotation_mat, translation_vec), dim=1) # shape (3,4)
    last_row= torch.tensor([[0,0,0,1]], dtype=torch.float32).to(rotation_mat.device)
    homo_mat = torch.cat((homo_mat, last_row), dim=0)  # shape (4,4)
    return homo_mat


def transform_point_torch(point_cloud, transformation_matrix):
    '''
    point cloud: shape (num_pts, 3)
    transformation_matrix: shape (4, 4)

    output: 
        shape (num_pts, 3)
    '''
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = torch.cat((point_cloud, torch.ones((point_cloud.shape[0], 1)).to(point_cloud.device)), dim=-1)
    # Apply the transformation matrix to each point
    transformed_points = torch.matmul(homogeneous_points, transformation_matrix.transpose(0,1).to(point_cloud.device))
    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]
    return transformed_points




class GraspPlanner(nn.Module):
    def __init__(self, euler_angles, translation_vec):
        '''
        
        optimize wrt euler angles and translation_vec
        '''
        super(GraspPlanner, self).__init__()
        self.euler_angles = nn.Parameter(torch.tensor(euler_angles).float()) #torch.randn((3,), dtype=torch.float32)
        self.translation_vec = nn.Parameter(torch.tensor(translation_vec).float()) #torch.randn((3,), dtype=torch.float32)
    
    def forward(self, query, object_pc, gripper_pc, stress_net, world_to_obj_homo_mat):
        '''
        gripper pc: constant # shape (num_pts, 5)
        object pc: constant # shape (num_pts, 5)
        query: shape (B, num_qrs, 5)
        world_to_obj_homo_mat: shape (4,4)
        '''
        grasp_homo_mat = pose_to_homo_mat(self.euler_angles, translation_vec=self.translation_vec)
        transformed_gripper_pc = transform_point_torch(gripper_pc[:, :3], grasp_homo_mat) # rotate and translate to desired gripper pose
        transformed_gripper_pc = transform_point_torch(transformed_gripper_pc[:, :3], world_to_obj_homo_mat) # transform from world to object frame
        transformed_gripper_pc = torch.cat((transformed_gripper_pc, gripper_pc[:, 3:]), dim=1) # shape (num_pts, 5); augment the transformed gripper pointcloud


        combined_pc = torch.cat((object_pc, transformed_gripper_pc), dim=0) # shape (num_pts*2, 5)

        B = query.shape[0] 
        num_queries = query.shape[1]
        pc = combined_pc.unsqueeze(0).expand(B, combined_pc.shape[0], combined_pc.shape[1]) # shape (B, num_pts*2, 5)
        pc = pc.permute(0,2,1).float()  # shape (B, 5, num_pts*2)  

        _, x_occ = stress_net(pc, query) # x_occ has shape (B*num_queries, 1)

        x_occ = x_occ.reshape(B, num_queries) # shape (B, num_queries)

        return x_occ




if __name__ == '__main__':
    num_pts = 1024
    num_qrs = 2000
    B = 30
    device = torch.device("cuda") # "cpu"

    init_euler_angles = [1, 1, 1]
    init_translation_vec = [1, 2, 1]

    object_pc = torch.randn((num_pts, 5)).to(device)
    gripper_pc = torch.randn((num_pts, 5)).to(device)
    query = torch.randn((B, num_qrs, 3)).to(device)
    target_query_occ = torch.ones((B, num_qrs)).to(device)
    world_to_obj_homo_mat = torch.randn((4, 4)).to(device)
    model = StressNet2(num_channels=5).to(device)
    model.load_state_dict(torch.load("/home/shinghei/Downloads/shinghei_stuff(1)/all_6polygon_open_gripper/epoch 193"))
    model.eval()
    
    grasp_planner = GraspPlanner(init_euler_angles, init_translation_vec).to(device)

    optimizer = optim.Adam(grasp_planner.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    start_time = timeit.default_timer()
    for epoch in range(0, 10):
        print(f"================ Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        
        optimizer.zero_grad()
        predicted_occ = grasp_planner(query, object_pc, gripper_pc, model, world_to_obj_homo_mat) # shape (B, num_queries)
        
        loss = nn.BCELoss()(predicted_occ, target_query_occ) #chamfer_distance_batched(query, object_pc[:, :3], predicted_classes) 

        loss.backward()
        optimizer.step()

        print("euler grad", grasp_planner.euler_angles.grad)
        print("trans vec grad", grasp_planner.translation_vec.grad)
        scheduler.step()
            
    
