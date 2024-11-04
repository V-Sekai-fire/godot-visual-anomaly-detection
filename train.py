
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import torch
from multiprocessing import freeze_support


def train():
    torch.set_float32_matmul_precision('high')

    datamodule = Folder(
        name="chibire.com",
        root="datasets/flux1-schnell-fp8/CharacterDesign-FluxV2/",
        normal_dir="normal",
        abnormal_dir="abnormal",
        test_split_mode=TestSplitMode.SYNTHETIC,
        image_size=(1024,1024),
    )
    datamodule.setup()
    model = Patchcore()
    engine = Engine(max_epochs=50)
    engine.fit(datamodule=datamodule, model=model)

if __name__ == '__main__':
    freeze_support()
    train()