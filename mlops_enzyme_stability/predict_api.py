from transformers import BertModel, BertTokenizer
from fastapi import UploadFile, File, FastAPI
from http import HTTPStatus
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from pydantic import BaseModel
from omegaconf import OmegaConf
from models.MLP import MyNeuralNet
import torch


class embeddingDict(BaseModel):
    data_file: str
    state_dict: str
    type: str

app = FastAPI()

# Model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
encoder = BertModel.from_pretrained("Rostlab/prot_bert")
encoder = encoder.to(device)
encoder.eval()


@app.post("/Sequence_predictor/")
async def predict_step(data: embeddingDict):
    if data.type == "sequence":
        embeddings = []
        file1 = open(f"{data.data_file}", 'r')
        Lines = file1.readlines()
        for line in Lines:
            seq = line.strip()
            seq = " ".join(list(seq))
            token = tokenizer(seq, return_tensors='pt')
            output = encoder(**token.to(device))
            embeddings.append(output[1])
            embeddings = torch.Tensor(embeddings)
    elif data.type == "embedding":
        embeddings = torch.load(f"{data.data_file}")
    else:
        raise ValueError("type must be either sequence or embedding")
    
    config = OmegaConf.load("config.yaml")
    model = MyNeuralNet(config)
    dataloader = DataLoader(TensorDataset(embeddings),
                            batch_size=config.hyperparameters.batch_size, shuffle=True)
    model = model.to(device)
    state_dict = torch.load(f"{data.state_dict}")
    model.load_state_dict(state_dict)

    trainer = Trainer()
    predictions = trainer.predict(model, dataloader)
    return predictions