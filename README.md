# DCMD-Net-track-defect-detection
unsupervised RGBD track defect detection

**SETUP**

Our project is implemented based on the architecture of the anomalib library.

Please configure the anomalib library environment:

pip install anomalib

**Training**

1.Under the "src/anomalib/models" folder of the project file, create a new folder named DCMD-Net;

2.Download the above-mentioned.py file and put them in this folder;

3.Modify the "_ _ init _ _.py" file：

import anomalib.models.DCMD-Net import DCMD-Net

__all__=[
...,
"DCMD-Net",
]

4.Training：

python tools/train.py --model DCMD-Net --config src/anomalib/models/DCMD-Net/fastener3dconfig.yaml

**Inferencing**

python tools/inference/lightning_inference.py --config your.yaml --weights your.ckpt --input your_dir --input_test you_depth_dir --output your_dir

