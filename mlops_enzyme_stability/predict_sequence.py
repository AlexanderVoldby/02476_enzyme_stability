from fastapi import FastAPI, HTTPException, File, BackgroundTasks
from pydantic import BaseModel
from typing import List
import os
import torch
import gcsfs
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

background_tasks = BackgroundTasks()

# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
# model = BertModel.from_pretrained("Rostlab/prot_bert")
# model = model.to(device)
# model.eval()

# Hyperparameters
config_file_path = os.path.join(os.getcwd(), "config.yaml")
config = OmegaConf.load(config_file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_model_and_tokenizer():
    # Model is set to eval mode by default
    try:
        tokenizer = BertTokenizer.from_pretrained(config.BERT_path + "pretrained_tokenizer", do_lower_case=False )
        model = BertModel.from_pretrained(config.BERT_path + "pretrained_model")
    except:
        print("Downloading pretrained model")
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model = BertModel.from_pretrained("Rostlab/prot_bert")
    return model, tokenizer

def add_spaces(x):
    return " ".join(list(x))

def encode_sequences(sequences):
    model, tokenizer = get_bert_model_and_tokenizer()
    embeddings = []
    for seq in tqdm(sequences, desc="Encoding sequences", total=len(sequences)):
        assert type(seq) is str, "Sequences must be strings"
        token = tokenizer(add_spaces(seq), return_tensors='pt')
        output = model(**token.to(device))
        embeddings.append(output[1].detach().cpu())
    
    return torch.stack(embeddings)

def predict(tensors):
    
    mlp_model = load_mlp()
    dataloader = DataLoader(tensors, batch_size=config.hyperparameters.batch_size, shuffle=False)
    trainer = Trainer()
    
    predictions = trainer.predict(mlp_model, dataloader)
    predictions_vector = torch.cat(predictions, dim=0)
    
    print(predictions_vector)
    return predictions_vector.numpy().tolist()

def load_mlp():
    checkpoint_path = f"{config.checkpoint_path}/{config.best_model_name}.ckpt"
    model = MyNeuralNet.load_from_checkpoint(checkpoint_path)
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


@app.post("/predict/")
async def make_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        amino_acid_sequences = request.data
        global encoded_sequences
        encoded_sequences = encode_sequences(amino_acid_sequences)

        predictions = predict(encoded_sequences)
        save_predictions_background(predictions, amino_acid_sequences, background_tasks)
        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)