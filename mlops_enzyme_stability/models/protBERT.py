from transformers import BertModel, BertTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.to(device)
model.eval()

tokenizer.save_pretrained("models/protBERT/tokenizer", from_pt=True)
model.save_pretrained("models/protBERT/model", from_pt=True) 
