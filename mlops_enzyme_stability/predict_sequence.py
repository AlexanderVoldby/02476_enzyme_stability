from fastapi import FastAPI, HTTPException, File, BackgroundTasks
from pydantic import BaseModel
from typing import List
import os
import torch
from transformers import BertTokenizer, BertModel 
from torch.utils.data import DataLoader
from mlops_enzyme_stability.models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
import csv

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list[str]

class PredictionResponse(BaseModel):
    predictions: List[float]

background_tasks = BackgroundTasks()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.to(device)
model.eval()


def predict(cfg, tensors, modelpath):

    model = load_model(cfg, modelpath)
    dataloader = DataLoader(tensors, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    trainer = Trainer()
    predictions = trainer.predict(model, dataloader)
    predictions_vector = torch.cat(predictions, dim=0)
    return predictions_vector.numpy().tolist()

def load_model(cfg, path):
    model = MyNeuralNet(cfg)
    model = MyNeuralNet.load_from_checkpoint(path)
    model.eval()
    return model

def save_predictions(predictions, sequences):
    # Ensure predictions is a numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    # Create the directory if it doesn't exist
    output_dir = 'reports/predictions'
    os.makedirs(output_dir, exist_ok=True)

    # Define the CSV file path with timestamp
    timestamp = str(datetime.now())[:-5]
    output_file_path = os.path.join(output_dir, f'sequence_predictions.csv')

    # Write predictions to CSV
    with open(output_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i, (aas, pred) in enumerate(zip(sequences, predictions)):
            writer.writerow([i, timestamp, aas, pred])

    print(f"Predictions saved to {output_file_path}")

def save_predictions_background(predictions, sequences, background_tasks: BackgroundTasks):
    # Run the save_predictions function in the background
    background_tasks.add_task(save_predictions, predictions, sequences)

def add_spaces(x):
    return " ".join(list(x))

def encode_sequences(sequences):

    embeddings = []
    for seq in tqdm(sequences, desc="Encoding sequences", total=len(sequences)):
        assert type(seq) is str, "Sequences must be strings"
        token = tokenizer(add_spaces(seq), return_tensors='pt')
        output = model(**token.to(device))
        embeddings.append(output[1].detach().cpu())
    
    return torch.stack(embeddings)


@app.post("/predict/", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    # try:
    cfg = OmegaConf.load("config.yaml")
    amino_acid_sequences = request.data
    encoded_sequences = encode_sequences(amino_acid_sequences)
    
    checkpoint_path = f"{cfg.checkpoint_path}/{cfg.best_model_name}.ckpt"
    predictions = predict(cfg, encoded_sequences, checkpoint_path)
    save_predictions_background(predictions, amino_acid_sequences, background_tasks)
    return {"predictions": predictions}

    # except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)