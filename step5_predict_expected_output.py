# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import torch
import argparse
import numpy as np

import mmpose
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.apis import init_model as init_pose_estimator

import warnings
warnings.filterwarnings('ignore')

from cldetection_utils import load_train_stack_data, remove_zero_padding


def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # 初始化模型，init_pose_estimator 函数内部已经加载了模型权重和开启了eval()模式
    pose_estimator = init_pose_estimator(config=config.config_file, checkpoint=config.load_weight_path, device=device)

    # 加载测试数据为Numpy矩阵形式
    stacked_image_array = load_train_stack_data(config.load_mha_path)

    # 所有图像的测试结果列表
    all_images_predict_keypoints_list = []

    # 开始测试模型进行测试
    with torch.no_grad():
        pose_estimator.eval()
        for i in range(np.shape(stacked_image_array)[0]):
            # 切片出一张图像出来
            image = np.array(stacked_image_array[i, :, :, :])

            # 预处理去除0填充部分
            image = remove_zero_padding(image)

            # 模型调用进行预测，内部已经包含了配置文件中的test_pipeline操作，内部已经进行配置好的预处理操作，直接丢图就好啦
            # 如果前面有一个粗定位的模型，只需要改变bboxes参数，传入检测框坐标即可
            predict_results = inference_topdown(model=pose_estimator, img=image, bboxes=None, bbox_format='xyxy')

            # 由于 MMPose 兼容考虑到一张图有多个bboxes，所以返回的结果是多个 bboxes的关键点预测结果，虽然挑战赛的bbox只有一个
            # 但我们还需要调用 merge_data_samples 对结果进行合并
            result_samples = merge_data_samples(predict_results)

            # 取出对应的关键点的预测结果, pred_instances.keypoints shape is (检测框数量，关键点数量，2)，我们就一个检测框，所以索引是0
            keypoints = result_samples.pred_instances.keypoints[0, :, :]

            keypoints_list = []
            for i in range(np.shape(keypoints)[0]):
                # 索引得到不同的关键点热图
                x0, y0 = keypoints[i, 0], keypoints[i, 1]
                keypoints_list.append([x0, y0])
            all_images_predict_keypoints_list.append(keypoints_list)

    # save as expected format JSON file
    json_dict = {'name': 'Orthodontic landmarks', 'type': 'Multiple points'}

    all_predict_points_list = []
    for image_id, predict_landmarks in enumerate(all_images_predict_keypoints_list):
        for landmark_id, landmark in enumerate(predict_landmarks):
            points = {'name': str(landmark_id + 1),
                      'point': [landmark[0], landmark[1], image_id + 1]}
            all_predict_points_list.append(points)
    json_dict['points'] = all_predict_points_list

    # version information
    major = 1
    minor = 0
    json_dict['version'] = {'major': major, 'minor': minor}

    # JSON dict to JSON string
    json_string = json.dumps(json_dict, indent=4)
    with open(config.save_json_path, "w") as f:
        f.write(json_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config file | 模型的配置文件
    parser.add_argument('--config_file', type=str, default='cldetection_configs/exp42_td-hm_hrnet-w32_udp-8xb64-250e-1024x1024_hm1024_KeypointMSELoss_sigma6_moreaug1_longer_dark_decode256.py')

    # data parameters | 数据文件路径和配置文件的路径
    parser.add_argument('--load_mha_path', type=str, default='./data/challenges/CLdetection2023/train_stack.mha')
    parser.add_argument('--save_json_path', type=str, default='./step6_docker_and_upload/test/expected_output.json')

    # model load dir path | 最好模型的权重文件路径
    parser.add_argument('--load_weight_path', type=str, default='MMPose-checkpoints/exp42/best_SDR 2.0mm_epoch_50.pth')

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)

    experiment_config = parser.parse_args()
    main(experiment_config)