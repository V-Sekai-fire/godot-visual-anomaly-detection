from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import torch
from multiprocessing import freeze_support
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def train():
    torch.set_float32_matmul_precision("medium")

    datamodule = Folder(
        name="chibfire_com_style",
        root="datasets/a-poses",
        normal_dir="normal",
        abnormal_dir="abnormal",
        train_batch_size=64,
        eval_batch_size=64,
        num_workers=1,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_ratio=0.2,
    )
    datamodule.setup()
    model = Dinomaly()

    early_stopping = EarlyStopping(
        monitor="train_loss_epoch",
        patience=5,
        mode="min",  # Set to 'min' to stop when the loss stops decreasing
    )

    model_checkpoint = ModelCheckpoint(
        monitor="train_loss_epoch",
        mode="min",  # Set to 'min' to save the model with the lowest loss
        dirpath="checkpoints",
        filename="best_model",
    )

    engine = Engine(
        devices="1",
        precision="bf16-mixed",
        max_epochs=500,
        callbacks=[early_stopping, model_checkpoint],
    )

    engine.fit(datamodule=datamodule, model=model)


if __name__ == "__main__":
    freeze_support()
    train()
