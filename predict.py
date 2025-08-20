
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
        root="datasets/a-poses",
        normal_dir="normal",
        normal_test_dir="test",
        train_batch_size=1,
        num_workers=1,
    )

    model = Dinomaly()

    engine = Engine(devices="1", precision="bf16-mixed", max_epochs=500)

    predictions = engine.predict(
        datamodule=datamodule,
        model=model,
        ckpt_path="results/Dinomaly/chibfire_com_style/latest/weights/lightning/model.ckpt",
    )
    print(prediction)
    
if __name__ == '__main__':
    freeze_support()
    interfence()
