
from anomalib.engine import Engine
from anomalib.models import EfficientAd # EfficientAd has better time and acurracy performance than the patchcore default model.

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
        image_size=1024,
        train_batch_size=1,
        num_workers=0, # 0 for windows slowdown
    )
    datamodule.setup()
    model = EfficientAd()
    engine = Engine(max_epochs=50)
    engine.fit(datamodule=datamodule, model=model)

if __name__ == '__main__':
    freeze_support()
    train()