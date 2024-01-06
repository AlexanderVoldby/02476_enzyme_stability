import torch
from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet

# Training hyper parameters
lr = 0.001
batch_size = 16


# Generate dataset and dataloader
X = torch.load("data/processed/train_tensors.pt")
y = torch.load("data/processed/train_target.pt")

trainset = TensorDataset(X, y)
dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)

model = MyNeuralNet()

