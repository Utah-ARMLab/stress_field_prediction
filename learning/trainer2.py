import torch
import torch.optim as optim
from model import StressNet2
from dataset_loader import StressPredictionDataset2
import os
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import socket
import timeit

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
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        pc = sample["pc"].to(device)
        query = sample["query"].to(device)
        target_stress = sample["stress"].to(device)
        target_occupancy = sample["occupancy"].to(device)
        
        num_qrs = query.shape[1]
        target_stress = target_stress.reshape(-1,1)
        target_occupancy = target_occupancy.reshape(-1,1)
        print(target_stress.shape, target_occupancy.shape)
        print(pc.shape, query.shape)

            
        optimizer.zero_grad()
        output = model(pc, query)

        predicted_classes = (output[1] >= 0.5).squeeze().int()
        batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
        correct += batch_correct

        # if occupancy = 1, combine both losses from stress and occupancy
        # if occupancy = 0, only use the loss from occupancy            
        loss_occ = nn.BCELoss()(output[1], target_occupancy) #* 5e7   # occupancy loss
                
        occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
        if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
            selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
            loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs])  # stress loss
        else:
            loss_stress = 0
                    
        loss_stress /= 5e7
        
        # print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}. Ratio: {loss_stress.item()/loss_occ.item():.3f}")     # ratio should be = ~1    
        loss = loss_occ + loss_stress   
        
        
        loss.backward()
        train_loss += loss_stress.item()   #loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
        if batch_idx % 100 == 0 or batch_idx == len(train_loader.dataset) - 1:  
            train_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
            train_accuracies.append(100.* batch_correct / pc.shape[0])
    
    print('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/len(train_loader.dataset)))  
    print(f"Occupancy correct: {correct}/{len(train_loader.dataset)}. Accuracy: {100.*correct/len(train_loader.dataset):.2f}%")

    logger.info('(Train set) Average stress loss: {:.3f}'.format(
                train_loss/len(train_loader.dataset)))  
    logger.info(f"Occupancy correct: {correct}/{len(train_loader.dataset)}. Accuracy: {100.*correct/len(train_loader.dataset):.2f}%")



def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
           
            pc = sample["pc"].to(device)
            query = sample["query"].to(device)
            target_stress = sample["stress"].to(device)
            target_occupancy = sample["occupancy"].to(device)

            num_qrs = query.shape[1]
            target_stress = target_stress.reshape(-1,1)
            target_occupancy = target_occupancy.reshape(-1,1)
            
            output = model(pc, query)
                    
            occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
            if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
                selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
                loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs], reduction='sum')  # stress loss
                test_loss += loss_stress.item()
           
            
            predicted_classes = (output[1] >= 0.5).squeeze().int()
            batch_correct = predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()
            correct += batch_correct

            # if batch_idx % 1 == 0 or batch_idx == len(test_loader.dataset) - 1:    
            test_stress_losses.append(loss_stress.item() / occupied_idxs.shape[0])
            test_accuracies.append(100.* batch_correct / pc.shape[0])      
                            

    test_loss /= len(test_loader.dataset)
    print('\n(Test set) Average stress loss: {:.3f}'.format(test_loss))
    print(f"Occupancy correct: {correct}/{len(test_loader.dataset)}. Accuracy: {100.*correct/len(test_loader.dataset):.2f}%\n")  
    logger.info('(Test set) Average stress loss: {:.3f}'.format(test_loss))
    logger.info(f"Occupancy correct: {correct}/{len(test_loader.dataset)}. Accuracy: {100.*correct/len(test_loader.dataset):.2f}%\n")      

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

    weight_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/run1"
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
    dataset = StressPredictionDataset2(dataset_path)
    dataset_size = 100  #len(os.listdir(dataset_path))
    batch_size = 256
    
    train_len = 399    #round(dataset_size*0.9)
    test_len = 1     #round(dataset_size*0.1)-1
    total_len = train_len + test_len
    
    # Generate random indices for training and testing without overlap
    indices = torch.randint(low=0, high=dataset_size, size=(total_len,))  #torch.randperm(dataset_size)
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
    

    model = StressNet2(num_channels=5).to(device)
    model.apply(weights_init)
      
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    start_time = timeit.default_timer()
    for epoch in range(0, 31):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        
        print(f"======== Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        # if epoch % 1 == 0:            
        #     torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))
            
        # saved_data = {"train_stress_losses": train_stress_losses, "test_stress_losses": test_stress_losses,
        #         "train_accuracies": train_accuracies, "test_accuracies": test_accuracies} 
        # with open(os.path.join(weight_path, f"saved_losses_accuracies.pickle"), 'wb') as handle:
        #     pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        