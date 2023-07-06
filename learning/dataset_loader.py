import torch
import os
from torch.utils.data import Dataset
import pickle     

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