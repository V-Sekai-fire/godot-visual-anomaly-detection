# godot-visual-anomaly-detection

https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html

Create a model that will help developers find bugs using visual anomaly detection.

`pip install anomalib`

```python
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import EfficientAd # EfficientAd has better time and acurracy performance than the patchcore default model.

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode

datamodule = Folder(
    name="hazelnut_toy",
    root="datasets/hazelnut_toy",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
)

datamodule.setup()

model = EfficientAd()
engine = Engine(max_epochs=5)

engine.fit(datamodule=datamodule, model=model)

engine.fit(datamodule=datamodule, model=model, ckpt_path="path/to/checkpoint.ckpt")
```

```python
datamodule = Folder(
    name="hazelnut_toy",
    root="datasets/hazelnut_toy",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
)

model = EfficientAd()

predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```
