import torch
from torch.utils.data import DataLoader, TensorDataset

train_tensors = torch.load("data/processed/train_tensors.pt")
train_labels = torch.load("data/processed/train_target.pt")
test_tensors = torch.load("data/processed/test_tensors.pt")
test_labels = torch.load("data/processed/test_target.pt")


def test_dataloader():
    trainset = TensorDataset(train_tensors, train_labels)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    assert len(trainloader) == 1812, "Train dataloader should have 1812 batches"
    
    testset = TensorDataset(test_tensors, test_labels)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)
    print(len(testloader))
    assert len(testloader) == 151, "Test dataloader should have 151 batches"
