import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule
import os
from google.cloud import storage
import io

# File containing the neural network class used as a regression tool on the encoded amino acid sequences


class MyNeuralNet(LightningModule):
    """Basic neural network class.

    Args:
        config: Hydra config object
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.bucket_name = config.data_path.replace("gs://", "").split("/")[0]

        self.batch_size = config.hyperparameters.batch_size
        self.lr = config.hyperparameters.lr
        self.num_workers = config.hyperparameters.num_workers
        self.mlp = nn.Sequential(
            nn.Linear(1024, config.hyperparameters.hidden1),
            nn.ReLU(),
            nn.Linear(config.hyperparameters.hidden1, config.hyperparameters.hidden2),
            nn.ReLU(),
            nn.Linear(config.hyperparameters.hidden2, 1),
        )
        self.criterion = self.configure_criterion()
        self.optimizer = self.configure_optimizers()

        # Save hyperparameters to use with wandb
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return torch.flatten(self.mlp(x))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step of the model.

        Args:
            batch: batch of data
            batch_idx: index of the batch

        Returns:
            Loss tensor

        """
        data, label = batch
        pred = self(data)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch
        return self(x)

    def configure_optimizers(self):
        """Optimizer configuration.

        Returns:
            Optimizer

        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_criterion(self):
        return torch.nn.MSELoss()

    def train_dataloader(self):
        # print(f"CWD: {os.getcwd()}") # TODO: Remove when done debugging
        storage_client = storage.Client(project="enzyme-stability-02476")
        bucket = storage_client.bucket(self.bucket_name)

        blob_target = bucket.blob("data/processed/train_target.pt")
        train_target_blob = blob_target.download_as_bytes()
        train_target_tensor = torch.load(io.BytesIO(train_target_blob))

        blob_tensors = bucket.blob("data/processed/train_tensors.pt")
        train_tensors_blob = blob_tensors.download_as_bytes()
        train_tensors_tensor = torch.load(io.BytesIO(train_tensors_blob))

        # TODO: UserWarning: Using a target size (torch.Size([16])) that is different to the input size (torch.Size([16, 1, 1])).
        # This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        trainset = TensorDataset(train_tensors_tensor, train_target_tensor)
        return DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    # def val_dataloader(self): #TODO: implement validation dataloader
    #     return DataLoader(...)

    def predict_dataloader(self):
        storage_client = storage.Client(project="enzyme-stability-02476")
        bucket = storage_client.bucket(self.bucket_name)

        # blob_target = bucket.blob("data/processed/test_target.pt")
        # test_target_blob = blob_target.download_as_bytes()
        # test_target_tensor = torch.load(io.BytesIO(test_target_blob))

        blob_tensors = bucket.blob("data/processed/test_tensors.pt")
        test_tensors_blob = blob_tensors.download_as_bytes()
        test_tensors_tensor = torch.load(io.BytesIO(test_tensors_blob))

        # testset = TensorDataset(test_tensors_tensor, test_target_tensor)
        # return DataLoader(testset, shuffle=False, batch_size=self.batch_size)
        return DataLoader(
            test_tensors_tensor, shuffle=False, batch_size=self.batch_size
        )
