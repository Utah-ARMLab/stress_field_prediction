import torch
import torch.optim as optim
from model import StressNet
from dataset_loader import StressPredictionDataset
import os
import torch.nn.functional as F
import torch.nn as nn
import random

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
            
        optimizer.zero_grad()
        output = model(pc, query)

        predicted_classes = (output[1] >= 0.5).squeeze().int()
        correct += predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()

        # if occupancy = 1, combine both losses from stress and occupancy
        # if occupancy = 0, only use the loss from occupancy            
        loss_occ = nn.BCELoss()(output[1], target_occupancy) #* 30   # occupancy loss
                
        occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
        if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
            selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
            loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs])  # stress loss
        else:
            loss_stress = 0
        
        print(f"Loss occ: {loss_occ.item():.3f}. Loss Stress: {loss_stress.item():.3f}")       
        loss = loss_occ + loss_stress   
        
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    print('====> Epoch: {} Average stress loss: {:.6f}'.format(
              epoch, train_loss/num_batch))  
    print(f"Occupancy correct: {correct}/{len(train_loader.dataset)}. Accuracy: {100.*correct/len(train_loader.dataset):.2f}%")




def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
           
            pc = sample["pc"].to(device)
            query = sample["query"].to(device)
            target_stress = sample["stress"].to(device)
            target_occupancy = sample["occupancy"].to(device)
            
            output = model(pc, query)
                    
            occupied_idxs = torch.where(target_occupancy == 1)[0] # find where the query points belongs to volume of the obbject (occupancy = 1)        
            if occupied_idxs.numel() > 0:  # Check if there are any occupied indices
                selected_occupied_output = torch.index_select(output[0], 0, occupied_idxs)  # torch.index_select selects specific elements from output[0] based on the indices in occupied_idxs
                loss_stress = F.mse_loss(selected_occupied_output, target_stress[occupied_idxs], reduction='sum')  # stress loss
                test_loss += loss_stress.item()
           
            
            predicted_classes = (output[1] >= 0.5).squeeze().int()
            # print(output[1])
            # print(predicted_classes)
            # print(target_occupancy.int().view_as(predicted_classes))
            correct += predicted_classes.eq(target_occupancy.int().view_as(predicted_classes)).sum().item()

                  

    test_loss /= len(test_loader.dataset)
    print('\n(Test set) Average stress loss: {:.6f}'.format(test_loss))
    print(f"Occupancy correct: {correct}/{len(test_loader.dataset)}. Accuracy: {100.*correct/len(test_loader.dataset):.2f}%\n")  
    

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

    dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/processed_data"
    dataset = StressPredictionDataset(dataset_path)
    dataset_size = len(os.listdir(dataset_path))
    train_len = round(dataset_size*0.9)
    test_len = round(dataset_size*0.1)-1
    total_len = train_len + test_len
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("dataset size: ", dataset_size)
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)


    model = StressNet(num_channels=5).to(device)  # simple conv1D
    model.apply(weights_init)

    weight_path = "/home/baothach/shape_servo_data/stress_field_prediction/weights/run1"
    os.makedirs(weight_path, exist_ok=True)
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(0, 151):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 2 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))

