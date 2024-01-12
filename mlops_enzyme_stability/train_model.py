from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import torch

# Training hyper parameters
# lr = 0.001
# epochs = 5
# batch_size = 16
# in_features = 1024
# hidden1 = 512
# hidden2 = 256
# out_features = 1

# def train(model, dataloader, lr, epochs):

#     optimizer = optim.Adam(params=model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for e in range(epochs):
#         for tensors, target in dataloader:
#             optimizer.zero_grad()
#             output = model(tensors)
#             # Optimize the RMSE since this is what they use in the competition
#             loss = torch.sqrt(criterion(target, output))
#             loss.backward()
#             optimizer.step()

#     torch.save("models/model_checkpoint.pt", model.state_dict())

if __name__ == "__main__":
    # Generate dataset and dataloader
    # X = torch.load("mlops_enzyme_stability/data/processed/train_tensors.pt")
    # y = torch.load("mlops_enzyme_stability/data/processed/train_target.pt")

    # model = MyNeuralNet(in_features=in_features,
    #                 hidden1=hidden1,
    #                 hidden2=hidden2,
    #                 out_features=out_features)

    # trainset = TensorDataset(X, y)
    # dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    # train(model, dataloader, lr=lr, epochs=epochs)

    # TODO: Hydra here
    config = OmegaConf.load("config.yaml")
    print(config) # TODO: Remove when done debugging
    
    # Wandb logger
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    
    # Train model
    model = MyNeuralNet(config)
    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback],
                      max_epochs=config.hyperparameters.epochs)
    trainer.fit(model)
    # Save state_dict for later use
    torch.save(model.state_dict(), "models/model_checkpoint.pt")

