
from anomalib.engine import Engine
from anomalib.models import EfficientAd # EfficientAd has better time and acurracy performance than the patchcore default model.

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode

datamodule = Folder(
    name="chibire.com",
    root="datasets/flux1-schnell-fp8/CharacterDesign-FluxV2",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
)

datamodule.setup()

model = EfficientAd()
engine = Engine(max_epochs=5)

engine.fit(datamodule=datamodule, model=model)
