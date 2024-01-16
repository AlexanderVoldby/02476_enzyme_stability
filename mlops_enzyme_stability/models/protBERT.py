from transformers import BertModel, BertTokenizer
import re

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

tokenizer.save_pretrained("models/protBERT/tokenizer", from_pt=True)
model.save_pretrained("models/protBERT/model", from_pt=True) 
