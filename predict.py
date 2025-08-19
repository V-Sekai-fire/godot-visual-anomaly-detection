
from anomalib.models import Dinomaly
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import torch
from multiprocessing import freeze_support

def interfence():
    torch.set_float32_matmul_precision('medium')
        
    datamodule = Folder(
        name="chibfire.com",
        root="datasets/flux1-schnell-fp8/CharacterDesign-FluxV2",
        normal_dir="abnormal",
        test_split_mode=TestSplitMode.SYNTHETIC,
    )

    model = Dinomaly()

    engine = Engine()

    predictions = engine.predict(
        datamodule=datamodule,
        model=model,
        ckpt_path="results/Dinomaly/chibfire_com_style/v2/weights/lightning/model.ckpt",
    )
    
    print(predictions)
    
if __name__ == '__main__':
    freeze_support()
    interfence()