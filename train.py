from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
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
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=1,
        test_split_mode=TestSplitMode.SYNTHETIC,  # Change this to create a validation set
        val_split_mode=ValSplitMode.SAME_AS_TEST,  # Add this to use the same split as the test set
        # You can also use ValSplitMode.SYNTHETIC and a ratio, e.g., val_split_ratio=0.1
    )
    datamodule.setup()
    model = Dinomaly()

    early_stopping = EarlyStopping(monitor="val_image_AUROC", patience=5, mode="max")

    model_checkpoint = ModelCheckpoint(
        monitor="val_image_AUROC",
        mode="max",
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
