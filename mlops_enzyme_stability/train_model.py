import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
import hydra

# Training hyper parameters
@hydra.train(config_name="basic_config.yaml")
def train(model, dataset, cfg):

    dataloader = DataLoader(dataset, shuffle=True, batch_size=cfg.hyperparameters.batch_size)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.hyperparameters.lr)
    criterion = nn.MSELoss()
    eps = cfg.hyperparameters.epochs
    for e in range(eps):
        print(f"Epoch {e+1}/{eps}")
        for tensors, target in dataloader:
            optimizer.zero_grad()
            output = torch.flatten(model(tensors))
            # Optimize the RMSE since this is what they use in the competition
            loss = torch.sqrt(criterion(target, output))
            loss.backward()
            optimizer.step()

    torch.save("models/model_checkpoint.pt", model.state_dict())

if __name__ == "__main__":
    # Generate dataset and dataloader
    X = torch.load("data/processed/train_tensors.pt")
    y = torch.load("data/processed/train_target.pt")

    model = MyNeuralNet()
    trainset = TensorDataset(X, y)
    
    train(model, trainset)






