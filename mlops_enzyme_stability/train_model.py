from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import wandb
import hydra


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../")
def main(config):
    # print(config)

    # Wandb login
    try:
        wandb.login(anonymous="allow", key=config.wandb_api_key, verify=True)
    except Exception:
        print("Wandb login failed")

    seed_everything(config.seed)
    wandb_logger = WandbLogger(
        log_model="all", project=config.project_name, name=config.runname
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        filename=config.runname,
        dirpath="gs://%s/%s"
        % (config.bucket_name, "models/MLP/checkpoints/%s" % config.runname),
        save_top_k=1,
    )
    model = MyNeuralNet(config)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.hyperparameters.epochs,
        # default_root_dir="gs://%s/%s"
        # % (
        #     config.gs_bucket_name,
        #     "models/MLP/checkpoints/%s/" % config.runname,
        # ),
    )
    trainer.fit(model)

    return


if __name__ == "__main__":
    main()
