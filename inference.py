
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import torch
from multiprocessing import freeze_support

def interfence():
    torch.set_float32_matmul_precision('high')
        
    datamodule = Folder(
        name="chibire.com",
        root="datasets/flux1-schnell-fp8/CharacterDesign-FluxV2",
        normal_dir="test",
        test_split_mode=TestSplitMode.SYNTHETIC,
    )

    model = EfficientAd()

    engine = Engine(max_epochs=5)

    predictions = engine.predict(
        datamodule=datamodule,
        model=model,
        ckpt_path="results/EfficientAd/chibire.com/latest/weights/lightning/model.ckpt",
    )
    
    print(predictions)
    
if __name__ == '__main__':
    freeze_support()
    interfence()