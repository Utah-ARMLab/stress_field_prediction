import torch.nn as nn
import torch
import torch.nn.functional as F
from pointconv_util_groupnorm import PointConvDensitySetAbstraction

class StressNet(nn.Module):
    '''
    Stress Prediction
    '''
    def __init__(self, num_channels, normal_channel=False):
        super(StressNet, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

            
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=num_channels+3+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        # FC layers for query point
        self.fc1_query = nn.Linear(3, 64)
        self.bn1_query = nn.GroupNorm(1, 64)
        self.fc2_query = nn.Linear(64, 128)
        self.bn2_query = nn.GroupNorm(1, 128)
        self.fc3_query = nn.Linear(128, 256)
        self.bn3_query = nn.GroupNorm(1, 256)

        # FC layers to predict stress
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.GroupNorm(1, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.GroupNorm(1, 128)
        self.fc3 = nn.Linear(128, 1)

        # FC layers to predict occupancy (0 or 1)
        self.fc1_occ = nn.Linear(512, 256)
        self.bn1_occ = nn.GroupNorm(1, 256) 
        self.fc2_occ = nn.Linear(256, 128)
        self.bn2_occ = nn.GroupNorm(1, 128)
        self.fc3_occ = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pc, query):
        # Set Abstraction layers
        B,C,N = pc.shape
        if self.normal_channel:
            l0_points = pc
            l0_xyz = pc[:,:3,:]
        else:
            l0_points = pc
            l0_xyz = pc[:,:3,:]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        x_pc = l3_points.view(B, 256)

        x_qr = F.relu(self.bn1_query(self.fc1_query(query)))
        x_qr = F.relu(self.bn2_query(self.fc2_query(x_qr)))
        x_qr = F.relu(self.bn3_query(self.fc3_query(x_qr)))
        
        x = torch.cat((x_pc, x_qr),dim=-1)
        
        x_stress = F.relu(self.bn1(self.fc1(x)))
        x_stress = F.relu(self.bn2(self.fc2(x_stress)))
        x_stress = self.fc3(x_stress)

        x_occ = F.relu(self.bn1_occ(self.fc1_occ(x)))
        x_occ = F.relu(self.bn2_occ(self.fc2_occ(x_occ)))
        x_occ = self.fc3_occ(x_occ)
        x_occ = self.sigmoid(x_occ)

        return x_stress, x_occ


if __name__ == '__main__':

    device = torch.device("cuda") # "cpu"
    num_pts = 1234
    num_batch = 8
    num_channels = 5
    
    pc = torch.randn((num_batch,num_channels,num_pts)).float().to(device)
    query = torch.randn((num_batch,3)).float().to(device)
    
    model = StressNet(num_channels).to(device)
    out = model(pc, query)
    print("out.shape:", out[0].shape, out[1].shape)