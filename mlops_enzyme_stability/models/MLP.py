import torch
# File containing the neural network class used as a regression tool on the encoded amino acid sequences

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int,
                 hidden1: int,
                 hidden2: int,
                 out_features: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, hidden1)
        self.l2 = torch.nn.Linear(hidden1, hidden2)
        self.l3 = torch.nn.Linear(hidden2, out_features)
        self.r = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l3(self.r(self.l2(self.r(self.l1(x)))))