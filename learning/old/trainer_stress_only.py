import torch
import torch.optim as optim
from model import StressNetStressOnly
from dataset_loader import StressPredictionDataset3
import os
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import socket
import timeit
import numpy as np
import sys
sys.path.append("../")
from utils.miscellaneous_utils import print_color


train_stress_losses = []
test_stress_losses = []
# train_occ_losses = []
# test_occ_losses = []
train_accuracies = []
test_accuracies = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    correct = 0
    total_num_qrs = 0
    total_occupied_qrs = 0

    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
            
        pc = sample["pc"].to(device)
        query = sample["query"].to(device)
        target_stress = sample["stress"].to(device)
        target_occupancy = sample["occupancy"].to(device)
               
        target_stress = target_stress.reshape(-1,1) # shape (total_num_qrs,1)
        target_occupancy = target_occupancy.reshape(-1,1) # shape (total_num_qrs,1)
        # num_queries = query.shape[1]
        # total_num_qrs = target_stress.shape[0]  # = 8*B*num_queries = total number of query points from 8 cams, B batches (B point clouds), num_queries each batch.

        pc = pc.view(-1, pc.shape[2], pc.shape[3])  # shape (B*8, 5, num_pts*2)
        query = query.view(-1, query.shape[2], query.shape[3])  # shape (B*8, num_queries, 3)

        # print(target_stress.shape, target_occupancy.shape)
        # print(pc.shape, query.shape)
            
        optimizer.zero_grad()
        output = model(pc, query)

        # if occupancy = 1, combine both losses from stress and occupancy
        # if occupancy = 0, only use the loss from occupancy            
                
        occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
        if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
            loss_stress = F.mse_loss(output[occupied_idxs], target_stress[occupied_idxs])  # stress loss
        else:
            loss_stress = 0
                    
        
        # print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}. Ratio: {loss_stress.item()/loss_occ.item():.3f}")     # ratio should be = ~1    
        loss = loss_stress 
        
        
        loss.backward()
        train_loss += loss_stress.item()   #loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            # print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}. Ratio: {loss_stress.item()/loss_occ.item():.3f}")     # ratio should be = ~1 
            
        
        if batch_idx % 100 == 0 or batch_idx == len(train_loader.dataset) - 1:  
            train_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
    
    print('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/num_batch))  

    logger.info('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/num_batch))  



def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct = 0
    total_num_qrs = 0
    total_occupied_qrs = 0
    num_batch = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            
            num_batch += 1
            
            pc = sample["pc"].to(device)
            query = sample["query"].to(device)
            target_stress = sample["stress"].to(device)
            target_occupancy = sample["occupancy"].to(device)

            target_stress = target_stress.reshape(-1,1) # shape (total_num_qrs,1)
            target_occupancy = target_occupancy.reshape(-1,1) # shape (total_num_qrs,1)


            pc = pc.view(-1, pc.shape[2], pc.shape[3])  # shape (B*8, 5, num_pts*2)
            query = query.view(-1, query.shape[2], query.shape[3])  # shape (B*8, num_queries, 3)
            
            
            output = model(pc, query)
                    
            occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
            if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
                loss_stress = F.mse_loss(output[occupied_idxs], target_stress[occupied_idxs], reduction="sum")  # stress loss
                test_loss += loss_stress.item()

            # print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(sample), len(test_loader.dataset),
            #     100. * batch_idx / len(test_loader), loss_stress.item()))           

            # if batch_idx % 1 == 0 or batch_idx == len(test_loader.dataset) - 1:    
            test_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
                            

    test_loss /= len(test_loader.dataset)
    print('\n(Test set) Average stress loss: {:.3f}'.format(test_loss))
    logger.info('(Test set) Average stress loss: {:.3f}\n'.format(test_loss))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    torch.manual_seed(2021)
    device = torch.device("cuda")

    weight_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/run4(stress_only)"
    os.makedirs(weight_path, exist_ok=True)
    
    logger = logging.getLogger(weight_path)
    logger.propagate = False    # no output to console
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(weight_path, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Machine: {socket.gethostname()}")
   
    dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/shinghei_data"
    dataset = StressPredictionDataset3(dataset_path)
    dataset_size = len(os.listdir(dataset_path))
    batch_size = 20     
    
    train_len = round(dataset_size*0.9)
    test_len = round(dataset_size*0.1)-1
    total_len = train_len + test_len
    
    # Generate random indices for training and testing without overlap
    # indices = torch.randint(low=0, high=dataset_size, size=(total_len,))  #torch.randperm(dataset_size)
    indices = np.arange(dataset_size)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:total_len]

    # Create Subset objects for training and testing datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)    
       
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)
    logger.info(f"Train len: {len(train_dataset)}")    
    logger.info(f"Test len: {len(test_dataset)}") 
    logger.info(f"Data path: {dataset.dataset_path}") 
    

    model = StressNetStressOnly(num_channels=5).to(device)
    model.apply(weights_init)
      
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    
    start_time = timeit.default_timer()
    for epoch in range(0, 61):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        
        print_color(f"================ Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 1 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))
            
        saved_data = {"train_stress_losses": train_stress_losses, "test_stress_losses": test_stress_losses,
                "train_accuracies": train_accuracies, "test_accuracies": test_accuracies} 
        with open(os.path.join(weight_path, f"saved_losses_accuracies.pickle"), 'wb') as handle:
            pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        