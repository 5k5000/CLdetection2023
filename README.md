# The Solution Repository for MICCAI CLDetection2023 by Team SUTD-VLG

![PDF](https://github.com/5k5000/CLdetection2023/blob/master/Pictures_for_Github_only/mainframework.png)
Our method will be made public on arxiv very soon.
## Performance

The online results on the public leaderboards could be viewed at [board1](https://cl-detection2023.grand-challenge.org/evaluation/challenge/leaderboard/) and [board2](https://cl-detection2023.grand-challenge.org/evaluation/testing/leaderboard/). (Algorithm Name: SUTD-VLG,  User Name: fivethousand).


![Online Result](https://github.com/5k5000/CLdetection2023/blob/master/Pictures_for_Github_only/Online%20Result.png)

Our method achieves 1st place ranking on three metrics and 3rd place on the remaining one.



## A step-by-step Tutorial

### 1. Conda Env Preparation
To build a compatible conda env, you only need to run the following lines one by one:
```
conda create -n LMD python=3.10
conda activate LMD
pip install -r requirements.txt
pip install -U openmim
cd mmpose_package/mmpose
pip install -e .
mim install mmengine
mim install "mmcv>=2.0.0"
pip install --upgrade numpy
```
To validate the effectiveness of the built conda env, you could run `step1_test_mmpose.py`. It will report the version of the installed mmpose package.

### 2. Download the Cldetection2023 dataset
As we do not have the right to forward the CLdetection dataset, interested researhers should follow this [website](https://cl-detection2023.grand-challenge.org/training-datasets/) to request it.
After being approved, you will have access to 2 files: train_stack.mha and train-gt.json. Then you could download them and place them under the `./data`.
After that , you will have the following data structure under `./data`:
```
.
├── train-gt.json
└── train_stack.mha
```


### 3. Convert to coco-style dataset
To make the dataset structure compatible with the MMPose package, you should convert the original dataset into a coco-style one with the provided script `step2_prepare_coco_dataset.py`.
```
python step2_prepare_coco_dataset.py --mha_file_path ./data/train_stack.mha --train_gt_path ./data/train-gt.json --image_save_dir ./data/preprocessed
```

It will generate the preprocessed dataset, together with the train.json, valid.json, test.json. Then, the `./data` directory will have the following file structure:
```
.
├── preprocessed
├── test.json
├── train-gt.json
├── train.json
├── train_stack.mha
└── valid.json
```

### 4. Train
Specify the config and the working directory, and run:
```
python step3_train_and_evaluation.py --config 'configs/CLdetection2023/srpose_s2.py' --work-dir './MMPose-checkpoints/CLdetection/srpose_s2'
```
Note that both the checkpoints and logs will be saved in the `--work-dir`.



### 5. Test
Test with the pretrained weights:
```
python step4_test_and_visualize.py --config 'configs/CLdetection2023/srpose_s2.py' --checkpoint './MMPose-checkpoints/CLdetection/srpose_s2/demo.pth'
```








## Acknowledgement
We would like to thank the [MICCAI CLDetection2023 organizers](https://cl-detection2023.grand-challenge.org/) for providing well-established [baselines](https://github.com/szuboy/CL-Detection2023) and their altruistic service for the contest. We appreciate all the contributors of the [MMPose](https://github.com/open-mmlab/mmpose) Package. We thank the authors of [SRPose](https://github.com/haonanwang0522/SRPose) for making their code public. 