import torch
from torch.utils.data import DataLoader, TensorDataset
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from datetime import datetime
import hydra
import os
import csv

class embeddingDict(BaseModel):
    data_file: str
    state_dict_file: str

app = FastAPI()

@hydra.main(version_base="1.3", config_name="config.yaml", config_path="./")
@app.post("/predict/")
def predict(cfg, data: embeddingDict, background_tasks: BackgroundTasks):
    # define checkpoint path and load model
    checkpoint_file = f"{data.state_dict_file}"
    checkpoint_path = os.path.join(cfg.checkpoint_path, checkpoint_file)
    print("Loading model")
    model = load_model(cfg, checkpoint_path)

    # Load test data
    test_tensors = torch.load(data.data_file)
    #test_target = torch.load("data/processed/test_target.pt")
    dataloader = DataLoader(test_tensors,
                           batch_size=cfg.hyperparameters.batch_size,
                           shuffle=False)


    trainer = Trainer()
    print("Calculating predictions")
    predictions = trainer.predict(model, dataloader)
    predictions_vector = torch.cat(predictions, dim=0)
    now = str(datetime.now())
    n = 0
    print("Saving predictions to database")
    for prediction in predictions_vector:
        background_tasks.add_task(add_to_database, now, prediction.item())
        n += 1

    return(f"{n} predictions have been saved to database {now}")


def load_model(cfg, path):
    model = MyNeuralNet(cfg)
    model = MyNeuralNet.load_from_checkpoint(path)
    model.eval()
    return model

def add_to_database(now: str, thermal_stability: float):
    """Simple function to add prediction to database."""
    with open("reports/predictions/predictions.csv", "a") as file:
        file.write(f"{now}, {thermal_stability}\n")

