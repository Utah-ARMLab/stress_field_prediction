import torch.nn as nn
import torch
import torch.nn.functional as F
from pointconv_util_groupnorm import PointConvDensitySetAbstraction


class PointCloudEncoder(nn.Module):
    '''
    Encode point cloud to latent space.
    '''
    def __init__(self, num_channels):
        super(PointCloudEncoder, self).__init__()
        
           
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=num_channels+3, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)


    def forward(self, pc):
        # Set Abstraction layers
        B,C,N = pc.shape
        l0_points = pc
        l0_xyz = pc[:,:3,:]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        x_pc = l3_points.view(B, 256)
        
        return x_pc

class PointCloudEncoderConv1D(nn.Module):
    '''
    Encode point cloud to latent space.
    '''
    def __init__(self, num_channels, embedding_size=256):
        super(PointCloudEncoderConv1D, self).__init__()
        
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=self.embedding_size, kernel_size=1)

        # batch norm
        self.bn1 = nn.GroupNorm(1, 64)
        self.bn2 = nn.GroupNorm(1, 128)
        self.bn3 = nn.GroupNorm(1, self.embedding_size)

    def forward(self, pc):
        #encoder
        x = F.relu(self.bn1(self.conv1(pc)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x_pc = x.view(-1, self.embedding_size)
        
        # print("Baobaobaobaobao")
        
        return x_pc


class QueryEmbedderOccupancyOnly(nn.Module):
    '''
    Stress Prediction
    '''
    def __init__(self):
        super(QueryEmbedderOccupancyOnly, self).__init__()
        
        # FC layers for query point
        self.fc1_query = nn.Linear(3, 64)
        self.bn1_query = nn.GroupNorm(1, 64)
        self.fc2_query = nn.Linear(64, 128)
        self.bn2_query = nn.GroupNorm(1, 128)
        self.fc3_query = nn.Linear(128, 256)
        self.bn3_query = nn.GroupNorm(1, 256)

        # FC layers to predict occupancy (0 or 1)
        self.fc1_occ = nn.Linear(512, 256)
        self.bn1_occ = nn.GroupNorm(1, 256) 
        self.fc2_occ = nn.Linear(256, 128)
        self.bn2_occ = nn.GroupNorm(1, 128)
        self.fc3_occ = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pc_embedding, query):
       
        x_qr = F.relu(self.bn1_query(self.fc1_query(query)))
        x_qr = F.relu(self.bn2_query(self.fc2_query(x_qr)))
        x_qr = F.relu(self.bn3_query(self.fc3_query(x_qr)))
        
        x = torch.cat((pc_embedding, x_qr),dim=-1)
        
        x_occ = F.relu(self.bn1_occ(self.fc1_occ(x)))
        x_occ = F.relu(self.bn2_occ(self.fc2_occ(x_occ)))
        x_occ = self.fc3_occ(x_occ)
        x_occ = self.sigmoid(x_occ)

        return x_occ 
    
class StressNetOccupancyOnly(nn.Module):
    '''
    Stress Prediction
    '''
    def __init__(self, num_channels, pc_encoder_type=PointCloudEncoderConv1D):
        super(StressNetOccupancyOnly, self).__init__()
        
        self.pc_encoder = pc_encoder_type(num_channels=num_channels)
        self.qr_embedder = QueryEmbedderOccupancyOnly()

    def forward(self, pc, query):
        """ 
        pc: shape (B, 5, num_points)
        query: shape (B, num_qrs, 3)
        """
        
        num_qrs = query.shape[1]
        
        x = self.pc_encoder(pc).squeeze() # shape (B, 256)

        x = x.unsqueeze(1).repeat(1, num_qrs, 1) # shape (B, num_qrs, 256)
        x_occ = self.qr_embedder(x.view(-1, 256), query.view(-1, 3)) # query.view(-1, 3): shape (B * num_qrs, 3)

        
        return x_occ


class QueryEmbedderSDF(nn.Module):
    '''
    Stress Prediction
    '''
    def __init__(self):
        super(QueryEmbedderSDF, self).__init__()
        
        # FC layers for query point
        self.fc1_query = nn.Linear(3, 64)
        self.bn1_query = nn.GroupNorm(1, 64)
        self.fc2_query = nn.Linear(64, 128)
        self.bn2_query = nn.GroupNorm(1, 128)
        self.fc3_query = nn.Linear(128, 256)
        self.bn3_query = nn.GroupNorm(1, 256)

        # FC layers to predict signed distance (scalar)
        self.fc1_sdf = nn.Linear(512, 256)
        self.bn1_sdf = nn.GroupNorm(1, 256) 
        self.fc2_sdf = nn.Linear(256, 128)
        self.bn2_sdf = nn.GroupNorm(1, 128)
        self.fc3_sdf = nn.Linear(128, 1)


    def forward(self, pc_embedding, query):
   
        x_qr = F.relu(self.bn1_query(self.fc1_query(query)))
        x_qr = F.relu(self.bn2_query(self.fc2_query(x_qr)))
        x_qr = F.relu(self.bn3_query(self.fc3_query(x_qr)))
        
        x = torch.cat((pc_embedding, x_qr),dim=-1)
        
        x_sdf = F.relu(self.bn1_sdf(self.fc1_sdf(x)))
        x_sdf = F.relu(self.bn2_sdf(self.fc2_sdf(x_sdf)))
        x_sdf = self.fc3_sdf(x_sdf)

        return x_sdf 
    
class StressNetSDF(nn.Module):
    '''
    Stress Prediction
    '''
    def __init__(self, num_channels, pc_encoder_type=PointCloudEncoderConv1D):
        super(StressNetSDF, self).__init__()
        
        self.pc_encoder = pc_encoder_type(num_channels=num_channels)
        self.qr_embedder = QueryEmbedderSDF()

    def forward(self, pc, query):
        """ 
        pc: shape (B, 5, num_points)
        query: shape (B, num_qrs, 3)
        """
        
        num_qrs = query.shape[1]
        
        x = self.pc_encoder(pc).squeeze() # shape (B, 256)

        x = x.unsqueeze(1).repeat(1, num_qrs, 1) # shape (B, num_qrs, 256)
        x_sdf = self.qr_embedder(x.view(-1, 256), query.view(-1, 3)) # query.view(-1, 3): shape (B * num_qrs, 3)

        
        return x_sdf





if __name__ == '__main__':

    device = torch.device("cuda") # "cpu"
    num_pts = 1234
    num_batch = 31
    num_channels = 5
    
    # pc = torch.randn((num_batch,num_channels,num_pts)).float().to(device)
    # query = torch.randn((num_batch,3)).float().to(device)
    
    # model = StressNet(num_channels).to(device)
    # out = model(pc, query)
    # print("out.shape:", out[0].shape, out[1].shape)
    
    pc = torch.randn((num_batch,num_channels,num_pts)).float().to(device)
    # query = torch.randn((num_batch, 1000, 3)).float().to(device)
    
    # model = StressNet2(num_channels).to(device)
    # out = model(pc, query)
    # print("out.shape:", out[0].shape, out[1].shape)
    
    pc_encoder = PointCloudEncoderConv1D(num_channels=5).to(device)
    pc_embedding = pc_encoder(pc)
    print("pc_embedding.shape:", pc_embedding.shape)