import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule

# File containing the neural network class used as a regression tool on the encoded amino acid sequences

class MyNeuralNet(LightningModule):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(config.in_features, config.hidden1)
        self.l2 = torch.nn.Linear(config.hidden1, config.hidden2)
        self.l3 = torch.nn.Linear(config.hidden2, config.out_features)
        self.r = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l3(self.r(self.l2(self.r(self.l1(x)))))
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step of the model.
        
        Args:
            batch: batch of data
            batch_idx: index of the batch

        Returns:
            Loss tensor

        """
        data, label = batch
        pred = self(data)
        loss = self.criterion(pred, label)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self): 
        """Optimizer configuration.
        
        Returns:
            Optimizer

        """
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def train_dataloader(self):
        return DataLoader(...)

    def val_dataloader(self):
        return DataLoader(...)

    def test_dataloader(self):
        return DataLoader(...)