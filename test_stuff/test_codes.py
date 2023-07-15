import torch


x = torch.tensor([[1,2,3],[4,5,6]]) # shape (B, 256)
print(x.shape)
print(x)

x = x.unsqueeze(1).repeat(1, 5, 1) # shape (B, num_qrs, 256)
print(x)