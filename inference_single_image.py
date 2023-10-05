# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import torch
import argparse
import numpy as np
from mmengine.config import Config
import mmpose
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.apis import init_model as init_pose_estimator

import warnings
warnings.filterwarnings('ignore')

import sys
import os  
# 当前文件目录
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))
print(os.path.dirname(file_path))


from cldetection_utils import load_train_stack_data, remove_zero_padding
import pandas as pd
import json
import cv2
from glob import glob


def main(config):


    config_file = Config.fromfile(config.config_file)

    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # 初始化模型，init_pose_estimator 函数内部已经加载了模型权重和开启了eval()模式
    pose_estimator = init_pose_estimator(config=config.config_file, checkpoint=config.load_weight_path, device=device)

    # 所有图像的测试结果列表
    all_images_predict_keypoints_list = []

    if config.mode == "val":
        data_dir = config_file.val_dataloader["dataset"]["data_root"]
        data_prefix = config_file.val_dataloader["dataset"]["data_prefix"]["img"]
        ann_file = config_file.val_dataloader["dataset"]["ann_file"]
        imgs_dir = os.path.join(data_dir,data_prefix)
        ann_file = os.path.join(data_dir,ann_file)
        data_template = os.path.join(imgs_dir,"{}.png")

    else:
        data_dir = config_file.test_dataloader["dataset"]["data_root"]
        data_prefix = config_file.test_dataloader["dataset"]["data_prefix"]["img"]
        ann_file = config_file.test_dataloader["dataset"]["ann_file"]
        imgs_dir = os.path.join(data_dir,data_prefix)
        ann_file = os.path.join(data_dir,ann_file)
        data_template = os.path.join(imgs_dir,"{}.png")

    
    num_of_classes = config_file.model["head"]["num_joints"]

    # 打开JSON文件
    with open(ann_file) as file:
        # 加载JSON数据
        data = json.load(file)
        data_info = data["images"]
        data = data["annotations"]

    total_sdr_list = []
    total_mre_list = []
    # 开始测试模型进行测试
    with torch.no_grad():
        pose_estimator.eval()
        for i in range(len(data)):
            this_data = data[i]
            id = this_data["image_id"]
            gt = this_data["keypoints"]
            size = "{},{}".format(data_info[i]["width"],data_info[i]["height"])
            spacing = data_info[i]["spacing"]
            image_path = data_template.format(id)
            image = cv2.imread(image_path)
            # 预处理去除0填充部分
            image = remove_zero_padding(image)      # 经过这一步后没有变化, 主办方提供的baseline里的操作

            
            gt_array = np.array(gt)
            gt_array = gt_array.reshape(num_of_classes,3)

            # # 模型调用进行预测，内部已经包含了配置文件中的test_pipeline操作，内部已经进行配置好的预处理操作，直接丢图就好啦
            # # 如果前面有一个粗定位的模型，只需要改变bboxes参数，传入检测框坐标即可
            predict_results = inference_topdown(model=pose_estimator, img=image, bboxes=None, bbox_format='xyxy')

            # # 由于 MMPose 兼容考虑到一张图有多个bboxes，所以返回的结果是多个 bboxes的关键点预测结果，虽然挑战赛的bbox只有一个
            # # 但我们还需要调用 merge_data_samples 对结果进行合并
            result_samples = merge_data_samples(predict_results)

            # 取出对应的关键点的预测结果, pred_instances.keypoints shape is (检测框数量，关键点数量，2)，我们就一个检测框，所以索引是0
            keypoints = result_samples.pred_instances.keypoints[0, :, :]

            keypoints_list = []
            image_level_sdr2 = []
            image_level_mre = []

            for i in range(np.shape(keypoints)[0]):
                # 索引得到不同的关键点热图
                x0, y0 = keypoints[i, 0], keypoints[i, 1]
                gt_x,gt_y = gt[i*3],gt[i*3+1]
            
                keypoints_list.append([x0, y0])
                MRE = np.sqrt((x0-gt_x)**2+(y0-gt_y)**2) * spacing
                sdr2 = 1 if MRE <=2 else 0
                
                print("image_id:{} pid: {} predict ({},{}) gt ({},{})) MRE: {}".format(id,i,x0,y0,gt_x,gt_y,MRE))    
                image_level_sdr2.append(sdr2)
                image_level_mre.append(MRE)
                total_sdr_list.append(sdr2)
                total_mre_list.append(MRE)
            
            image_level_sdr2 = np.mean(image_level_sdr2)
            image_level_mre = np.mean(image_level_mre)

    mean_sdr = np.mean(total_sdr_list)
    mean_mre = np.mean(total_mre_list)
    std_mre = np.std(total_mre_list)
    print("mean_sdr: {} mean_mre: {} std_mre: {}".format(mean_sdr,mean_mre,std_mre))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config file | 模型的配置文件
    parser.add_argument('--config_file', type=str, default='configs/CLdetection2023/srpose_s2.py')

    # model load dir path | 模型的权重文件路径
    parser.add_argument('--load_weight_path', type=str, default='MMPose-checkpoints/CLdetection/srpose_s2/demo.pth')

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)
    # val or test
    parser.add_argument('--mode', type=str, default="val")

    experiment_config = parser.parse_args()
    main(experiment_config)