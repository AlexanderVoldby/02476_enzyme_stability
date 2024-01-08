import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet

# Training hyper parameters
lr = 0.001
epochs = 5
batch_size = 16
in_features = 1024
hidden1 = 512
hidden2 = 256
out_features = 1

def train(model, dataloader, lr, epochs):

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for tensors, target in dataloader:
            optimizer.zero_grad()
            output = model(tensors)
            # Optimize the RMSE since this is what they use in the competition
            loss = torch.sqrt(criterion(target, output))
            loss.backward()
            optimizer.step()

    torch.save("models/model_checkpoint.pt", model.state_dict())

if __name__ == "__main__":
    # Generate dataset and dataloader
    X = torch.load("mlops_enzyme_stability/data/processed/train_tensors.pt")
    y = torch.load("mlops_enzyme_stability/data/processed/train_target.pt")

    model = MyNeuralNet(in_features=in_features,
                    hidden1=hidden1,
                    hidden2=hidden2,
                    out_features=out_features)
    
    trainset = TensorDataset(X, y)
    dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    train(model, dataloader, lr=lr, epochs=epochs)






