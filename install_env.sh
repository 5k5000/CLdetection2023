conda create -n LMD python=3.10
conda activate LMD
pip install -r requirements.txt
pip install -U openmim
cd mmpose_package/mmpose
pip install -e .
mim install mmengine
mim install "mmcv>=2.0.0"
pip install --upgrade numpy

