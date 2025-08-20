
from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import torch
from multiprocessing import freeze_support


def train():
    torch.set_float32_matmul_precision('medium')

    datamodule = Folder(
        name="chibfire_com_style",
        root="datasets/t-pose-a-poses",
        normal_dir="normal",
        abnormal_dir="abnormal",
        test_split_mode=TestSplitMode.SYNTHETIC,
        train_batch_size=1,
        num_workers=1,
    )
    datamodule.setup()
    model = Dinomaly()
    engine = Engine(strategy="ddp", accelerator="gpu", devices="1", precision="bf16-mixed", max_epochs=500)
    engine.fit(datamodule=datamodule, model=model)

if __name__ == '__main__':
    freeze_support()
    train()
