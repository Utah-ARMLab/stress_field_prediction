import torch
import torch.optim as optim
from model import StressNet2
from dataset_loader import StressPredictionObjectFrameDataset, StressPredictionDataset3
import os
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import socket
import timeit
import numpy as np
import random
import sys
sys.path.append("../")
from utils.miscellaneous_utils import print_color

### Log for loss/accuracy plotting later
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

        pc = pc.view(-1, pc.shape[-2], pc.shape[-1])  # shape (B*8, 5, num_pts*2)
        query = query.view(-1, query.shape[-2], query.shape[-1])  # shape (B*8, num_queries, 3)

        # print(target_stress.shape, target_occupancy.shape)
        # print(pc.shape, query.shape)

            
        optimizer.zero_grad()
        output = model(pc, query)

        predicted_classes = (output[1] >= 0.5).squeeze().int()
        batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
        correct += batch_correct
        total_num_qrs += target_stress.shape[0]

        # if occupancy = 1, combine both losses from stress and occupancy
        # if occupancy = 0, only use the loss from occupancy            
        loss_occ = nn.BCELoss()(output[1], target_occupancy) # occupancy loss
                
        occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
        if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
            selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
            loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs])  # stress loss
        else:
            loss_stress = 0
                    

        loss_occ *= 65  # balance the two stress components 65 85
        
        print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}. Ratio stress/occ: {loss_stress.item()/loss_occ.item():.3f}")     # ratio should be = ~1    
        loss = loss_occ + loss_stress   
        
        
        loss.backward()
        train_loss += loss_stress.item()   #loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}. Ratio stress/occ: {loss_stress.item()/loss_occ.item():.3f}")     # ratio should be = ~1 
            
        
        if batch_idx % 10 == 0 or batch_idx == len(train_loader.dataset) - 1:  
            train_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
            train_accuracies.append(100.* batch_correct / output[0].shape[0])
    
    print('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/num_batch))  
    print(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%")

    logger.info('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/num_batch))  
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
            target_stress = sample["stress"].to(device)
            target_occupancy = sample["occupancy"].to(device)

            target_stress = target_stress.reshape(-1,1) # shape (total_num_qrs,1)
            target_occupancy = target_occupancy.reshape(-1,1) # shape (total_num_qrs,1)
            # num_queries = query.shape[1]
            # total_num_qrs = target_stress.shape[0]  # = 8*B*num_queries = total number of query points from 8 cams, B batches (B point clouds), num_queries each batch.

            pc = pc.view(-1, pc.shape[-2], pc.shape[-1])  # shape (B*8, 5, num_pts*2)
            query = query.view(-1, query.shape[-2], query.shape[-1])  # shape (B*8, num_queries, 3)
            
            
            output = model(pc, query)
                    
            occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
            total_occupied_qrs += occupied_idxs.shape[0]
            if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
                selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
                loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs])  # stress loss
                test_loss += loss_stress.item()
           
            
            predicted_classes = (output[1] >= 0.5).squeeze().int()
            batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
            correct += batch_correct
            total_num_qrs += target_stress.shape[0]

            # if batch_idx % 1 == 0 or batch_idx == len(test_loader.dataset) - 1:    
            test_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
            test_accuracies.append(100.* batch_correct / output[0].shape[0])      
                            

    test_loss /= len(test_loader.dataset)
    print('\n(Test set) Average stress loss: {:.3f}'.format(test_loss))
    print(f"Occupancy correct: {correct}/{total_num_qrs}. Accuracy: {100.*correct/total_num_qrs:.2f}%\n")  
    logger.info('(Test set) Average stress loss: {:.3f}'.format(test_loss))
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
    random.seed(2021)
    device = torch.device("cuda")

    weight_path = \
        "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness/weights/all_6polygon"
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
   
    dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
    gripper_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
    object_partial_pc_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
    object_names = [f"6polygon0{j}" for j in [3,4,5,6,7,8]]     # [3,4,5,6,7,8]

    # dataset = StressPredictionDataset3(dataset_path, gripper_pc_path, object_partial_pc_path)
    dataset = StressPredictionObjectFrameDataset(dataset_path, gripper_pc_path, object_partial_pc_path, object_names, joint_training=True)
    dataset_size = len(dataset)
    batch_size = 250     # 30     
    
    train_len = round(dataset_size*0.9)
    test_len = round(dataset_size*0.1)
    total_len = train_len + test_len
    
    # Generate random indices for training and testing without overlap
    indices = np.arange(dataset_size)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:total_len]


    # Create Subset objects for training and testing datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)    
       
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Total dataset size: ", len(dataset))
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)
    logger.info(f"Train len: {len(train_dataset)}")    
    logger.info(f"Test len: {len(test_dataset)}") 
    logger.info(f"Data path: {dataset.dataset_path}") 
    

    model = StressNet2(num_channels=5).to(device)
    model.apply(weights_init)
      
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    
    start_time = timeit.default_timer()
    for epoch in range(0, 101):     # For 6 6polygon objects, 8 transformed partial pcs, batch size 30, RTX 3090Ti, it takes ~6.8 hours to train 50 epochs.
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        
        print_color(f"================ Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        logger.info(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 1 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))
            
        saved_data = {"train_stress_losses": train_stress_losses, "test_stress_losses": test_stress_losses,
                "train_accuracies": train_accuracies, "test_accuracies": test_accuracies} 
        with open(os.path.join(weight_path, f"saved_losses_accuracies.pickle"), 'wb') as handle:
            pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
            
""" 
Training tips:
1) Final (train) average stress loss must be around 0.1~0.2
2) Occupancy accuracy (for both train and test) must be >= 94%. Sometimes it has to reach ~98% in order for the model to work.

"""
            
        