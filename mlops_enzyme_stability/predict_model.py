import torch
from torch.utils.data import DataLoader, TensorDataset
from models.MLP import MyNeuralNet
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

def predict():
    """
    Stuff needed for predicition:
    - path to data
    - path to model checkpoint
    - Determine whether data is an AA-sequence or an embedding of an AA-sequence
    - Embed AA-sequence if needed
    - Make predictions
    - Save predictions to file
    """
    # Load config
    config = OmegaConf.load("config.yaml")
    # Load model
    model = MyNeuralNet(config)
    state_dict = torch.load("models/model_checkpoint.pt")

    # Load config
    config = OmegaConf.load("config.yaml")
    # Load model
    model = MyNeuralNet(config)
    state_dict = torch.load("models/model_checkpoint.pt")

    # Load test data
    test_tensors = torch.load("data/processed/test_tensors.pt")
    test_target = torch.load("data/processed/test_target.pt")
    dataloader = DataLoader(TensorDataset(test_tensors, test_target),
                            batch_size=config.hyperparameters.batch_size, shuffle=True)
    model.load_state_dict(state_dict)

    trainer = Trainer()
    predictions = trainer.predict(model, dataloader)
    print(predictions)
