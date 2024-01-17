from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import wandb
import hydra
from google.cloud import storage

# Wandb login YOLO
try:
    wandb.login(
        anonymous="allow", key="8d8198f8b41c68eed39ef9021f8bea9633eb2f6e", verify=True
    )
except Exception:
    print("Wandb login failed")


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="./")
def main(config):
    print(config)
    seed_everything(config.seed)
    wandb_logger = WandbLogger(log_model="all")

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        filename=config.runname,
        dirpath=config.checkpoint_path,
        save_top_k=1,
    )
    model = MyNeuralNet(config)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.hyperparameters.epochs,
    )
    trainer.fit(model)

    # save model into te¨he GCS bucket
    storage_client = storage.Client(project=config.gs_project_name)

    model_name = config.runname + ".ckpt"
    gcs_model_path = "models/MLP/%s" % model_name
    local_model_path = config.checkpoint_path + config.runname + ".ckpt"

    bucket_name = config.gs_bucket_name
    bucket = storage_client.bucket(bucket_name)
    blob_model_dir = bucket.blob(gcs_model_path)
    blob_model_dir.upload_from_filename(local_model_path)

    return


if __name__ == "__main__":
    main()
