import torch
from torch.utils.data import DataLoader, TensorDataset
from mlops_enzyme_stability.models.MLP import MyNeuralNet

batch_size = 8
in_features = 1024
hidden1 = 512
hidden2 = 256
out_features = 1

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == "__main__":
    # Load test data
    test_tensors = torch.load("data/processed/...")
    test_target = torch.load("data/processed/...")
    dataloader = DataLoader(TensorDataset(test_tensors, test_target), batch_size=batch_size, shuffle=True)

    # Load model
    model = MyNeuralNet(in_features=in_features,
                        hidden1=hidden1,
                        hidden2=hidden2,
                        out_features=out_features)
    
    state_dict = torch.load("models/model_checkpoint.pt")
    model.load_state_dict(state_dict)

    # Make predictions
    pred = predict(model, dataloader)

