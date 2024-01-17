import torch
from torch.utils.data import DataLoader, TensorDataset
from mlops_enzyme_stability.models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
import hydra
import os
import csv
from google.cloud import storage
import io


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
    # define checkpoint path and load model
    storage_client = storage.Client(cfg.gs_project_name)
    bucket = storage_client.bucket(cfg.gs_bucket_name)

    blob_tensors = bucket.blob("models/MLP/%s.ckpt" % cfg.runname)
    st_dict_tensors_blob = blob_tensors.download_as_bytes()
    st_dict_tensors_tensor = torch.load(io.BytesIO(st_dict_tensors_blob))
    print(type(st_dict_tensors_tensor))
    # checkpoint_file = f"{cfg.runname}.ckpt"
    # checkpoint_path = os.path.join(cfg.checkpoint_path, checkpoint_file)

    model = load_model(cfg, st_dict_tensors_tensor)  # checkpoint_path)

    trainer = Trainer()
    # predictions = trainer.predict(model, dataloader)
    predictions = trainer.predict(model)
    predictions_vector = torch.cat(predictions, dim=0)
    return predictions_vector


def load_model(cfg, st_dict):
    model = MyNeuralNet(cfg)
    model.load_state_dict(
        state_dict=st_dict["state_dict"]
    )  # .load_from_checkpoint(path)
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
