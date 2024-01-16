from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import wandb
import hydra
import os

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

    return


if __name__ == "__main__":
    main()
