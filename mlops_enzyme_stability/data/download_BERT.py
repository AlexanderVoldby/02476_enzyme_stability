from transformers import BertModel, BertTokenizer
import hydra

#@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../../")
def download_pretrained_model():
    try:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        tokenizer.save_pretrained("mlops_enzyme_stability/models/protBERT/pretrained_tokenizer")
        model.save_pretrained("mlops_enzyme_stability/models/protBERT/pretrained_model")
    except:
        print("Failed to download pretrained model")
    
if __name__ == "__main__":
    download_pretrained_model()
    