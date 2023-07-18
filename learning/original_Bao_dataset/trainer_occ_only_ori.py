import torch
import torch.optim as optim
from model_ori import StressNetSDF, StressNetOccupancyOnly
from dataset_loader_ori import StressPredictionDataset
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
sys.path.append("../../")
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
        
        if train_sdf:
            target_signed_distance = sample["signed_distance"].to(device)              
            target_signed_distance = target_signed_distance.reshape(-1,1) # shape (total_num_qrs,1)            
        else:    
            target_occupancy = sample["occupancy"].to(device)              
            target_occupancy = target_occupancy.reshape(-1,1) # shape (total_num_qrs,1)


        pc = pc.view(-1, pc.shape[-2], pc.shape[-1])  # shape (B*8, 5, num_pts*2)
        query = query.view(-1, query.shape[-2], query.shape[-1])  # shape (B*8, num_queries, 3)

        print(target_occupancy.shape)
        print(pc.shape, query.shape)

            
        optimizer.zero_grad()
        output = model(pc, query)

        if not train_sdf:
            predicted_classes = (output >= 0.5).squeeze().int()
            batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
            correct += batch_correct
            total_num_qrs += target_occupancy.shape[0]

        if train_sdf:
            loss = F.mse_loss(output, target_signed_distance)
        else:
            loss = nn.BCELoss()(output, target_occupancy)   # occupancy loss

        train_loss += loss.item()   #loss.item()
        
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
       
        if batch_idx % 10 == 0 or batch_idx == len(train_loader.dataset) - 1:  
            train_stress_losses.append(loss.item())
            if not train_sdf:
                train_accuracies.append(100.* batch_correct / target_occupancy.shape[0])
    
    
    print('(Train set) Average stress loss: {:.6f}'.format(
                train_loss/num_batch))  
    if not train_sdf:
        print(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%")

    logger.info('(Train set) Average stress loss: {:.6f}'.format(
                train_loss/num_batch))  
    if not train_sdf:
        logger.info(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%")



def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct = 0
    total_num_qrs = 0
    total_occupied_qrs = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
           
            pc = sample["pc"].to(device)
            query = sample["query"].to(device)
            
            if train_sdf:
                target_signed_distance = sample["signed_distance"].to(device)              
                target_signed_distance = target_signed_distance.reshape(-1,1) # shape (total_num_qrs,1)            
            else:    
                target_occupancy = sample["occupancy"].to(device)              
                target_occupancy = target_occupancy.reshape(-1,1) # shape (total_num_qrs,1)


            pc = pc.view(-1, pc.shape[-2], pc.shape[-1])  # shape (B*8, 5, num_pts*2)
            query = query.view(-1, query.shape[-2], query.shape[-1])  # shape (B*8, num_queries, 3)
            
            
            output = model(pc, query)
                    
            if train_sdf:
                loss = F.mse_loss(output, target_signed_distance)
            else:
                loss = nn.BCELoss()(output, target_occupancy)   # occupancy loss    
                 
            test_loss += loss.item()
           
            if not train_sdf:
                predicted_classes = (output >= 0.5).squeeze().int()
                batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
                correct += batch_correct
                total_num_qrs += target_occupancy.shape[0]

            # if batch_idx % 1 == 0 or batch_idx == len(test_loader.dataset) - 1:    
            test_stress_losses.append(loss.item())
            if not train_sdf:
                test_accuracies.append(100.* batch_correct / target_occupancy.shape[0])      
                            

    test_loss /= len(test_loader.dataset)
    print('\n(Test set) Average stress loss: {:.6f}'.format(test_loss))
    if not train_sdf:
        print(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%\n")  
    logger.info('(Test set) Average stress loss: {:.6f}'.format(test_loss))
    if not train_sdf:
        logger.info(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%\n")   


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

    weight_path = "/home/baothach/shape_servo_data/stress_field_prediction/weights/cuboid01_2"
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
   
    # dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data_cuboid01"
    dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/shinghei_data_cuboid01"
    dataset = StressPredictionDataset(dataset_path)
    dataset_size = len(os.listdir(dataset_path))
    batch_size = 150     
    
    train_len = 1#round(dataset_size*0.9)
    test_len = 1#round(dataset_size*0.1)-1
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
    
    train_sdf = False
    if train_sdf:
        model = StressNetSDF(num_channels=5).to(device)
    else:
        model = StressNetOccupancyOnly(num_channels=5).to(device)
        
    model.apply(weights_init)
      
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    
    start_time = timeit.default_timer()
    for epoch in range(0, 151):
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
            
        