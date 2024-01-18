import io

import torch
from torch.utils.data import DataLoader, TensorDataset
from mlops_enzyme_stability.models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
import hydra
import os
import csv
from google.cloud import storage


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../")
def main(cfg):
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
    predictions = predict(cfg)
    save_predictions(predictions)


def predict(cfg):
    # define checkpoint path and load model
    # "gs://dtu-mlops-storage/models/MLP/checkpoints"
    checkpoint_path = f"{cfg.checkpoint_path}/{cfg.best_model_name}.ckpt"

    # model = MyNeuralNet(cfg)
    model = MyNeuralNet.load_from_checkpoint(checkpoint_path)

    trainer = Trainer()
    # predictions = trainer.predict(model, dataloader)
    predictions = trainer.predict(model)
    predictions_vector = torch.cat(predictions, dim=0)
    return predictions_vector


def load_model(cfg, path):
    model = MyNeuralNet.load_from_checkpoint(path)
    model.eval()
    return model


def save_predictions(predictions):
    # Ensure predictions is a numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    # Create the directory if it doesn't exist
    output_dir = "reports/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Define the CSV file path
    output_file_path = os.path.join(output_dir, "predictions.csv")

    # Write predictions to CSV
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for pred in predictions:
            writer.writerow([pred])  # Write each prediction in its own row

    print(f"Predictions saved to {output_file_path}")


if __name__ == "__main__":
    main()
