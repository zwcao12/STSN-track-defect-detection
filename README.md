# STSN-track-defect-detection
unsupervised RGBD track defect detection
SETUP
Our project is implemented based on the architecture of the anomalib library.
Please configure the anomalib library environment:
pip install anomalib
Training
1.Under the "src/anomalib/models" folder of the project file, create a new folder named STSN;
2.Download the above-mentioned.py file and put them in this folder;
3.Modify the __init__.py file：
import anomalib.models.STSN import STSN
__all__=[
...,
"STSN",
]
4.Training：
python tools/train.py --model STSN --config src/anomalib/models/STSN/fastener3dconfig.yaml
Inferencing
