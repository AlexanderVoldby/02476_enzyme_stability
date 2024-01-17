import io

import torch
from torch.utils.data import DataLoader, TensorDataset
from mlops_enzyme_stability.models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
import hydra
import os
import csv
from google.cloud import storage


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="./")
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
    model = load_model(cfg)

    trainer = Trainer()
    # predictions = trainer.predict(model, dataloader)
    predictions = trainer.predict(model)
    predictions_vector = torch.cat(predictions, dim=0)
    return predictions_vector


def load_model(cfg):
    # define checkpoint path and load model
    storage_client = storage.Client(project=cfg.gs_project_name)
    bucket = storage_client.bucket(cfg.gs_bucket_name)

    blob = bucket.blob("models/MLP/checkpoints/%s/%s.ckpt" % (cfg.runname, cfg.runname))
    blob_bin = blob.download_as_bytes()
    model_state_dict = torch.load(io.BytesIO(blob_bin))["state_dict"]

    model = MyNeuralNet(cfg)
    model.load_state_dict(state_dict=model_state_dict)
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
