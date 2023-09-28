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
import pandas as pd
import json
import cv2

def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # 初始化模型，init_pose_estimator 函数内部已经加载了模型权重和开启了eval()模式
    pose_estimator = init_pose_estimator(config=config.config_file, checkpoint=config.load_weight_path, device=device)

    # 所有图像的测试结果列表
    all_images_predict_keypoints_list = [] # ./data/challenges/CLdetection2023/preprocessed
    
    
    dataset_root = "./data/challenges/CLdetection2023/preprocessed"
    dataset_valid_json = "./data/challenges/CLdetection2023/valid.json"
    dataset_test_json = "./data/challenges/CLdetection2023/test.json"
    if config.mode == "val":
        dataset_json = dataset_valid_json
    else:
        dataset_json = dataset_test_json
    data_template = "./data/challenges/CLdetection2023/preprocessed/{}.png"
    
    dataframe_title = ["image_id"]
    for i in range(38):
        dataframe_title.append("p"+str(i+1)+"_sdr2")
        dataframe_title.append("p"+str(i+1)+"_MRE")
    
    fg_dataframe = pd.DataFrame(columns=dataframe_title)
    

    # 打开JSON文件
    with open(dataset_json) as file:
        # 加载JSON数据
        data = json.load(file)
        print(data)
        print(len(data))
        data_info = data["images"]
        data = data["annotations"]
        print(data)
        
        # print(len(data))
        
    total_sdr_list = []
    total_mre_list = []
    # 开始测试模型进行测试
    with torch.no_grad():
        pose_estimator.eval()
        for i in range(len(data)):
            # 切片出一张图像出来
            
            # image = np.array(stacked_image_array[i, :, :, :])
            this_data = data[i]
            id = this_data["image_id"]
            gt = this_data["keypoints"]
            size = "{},{}".format(data_info[i]["width"],data_info[i]["height"])
            spacing = data_info[i]["spacing"]
            image_path = data_template.format(id)
            image = cv2.imread(image_path)
            print(image.shape)
            # 预处理去除0填充部分
            image = remove_zero_padding(image)      # 经过这一步后没有变化
            print(image.shape)
            
            
            gt_array = np.array(gt)
            gt_array = gt_array.reshape(38,3)
            x_min = np.min(gt_array[:,0])
            x_max = np.max(gt_array[:,0])
            y_min = np.min(gt_array[:,1])
            y_max = np.max(gt_array[:,1])
            bbox = [x_min-20,y_min-20,x_max+20,y_max+ 20]
            bbox = np.array(bbox)
            bbox = bbox.reshape(1,4)


            # # 模型调用进行预测，内部已经包含了配置文件中的test_pipeline操作，内部已经进行配置好的预处理操作，直接丢图就好啦
            # # 如果前面有一个粗定位的模型，只需要改变bboxes参数，传入检测框坐标即可
            predict_results = inference_topdown(model=pose_estimator, img=image, bboxes=None, bbox_format='xyxy')

            # # 由于 MMPose 兼容考虑到一张图有多个bboxes，所以返回的结果是多个 bboxes的关键点预测结果，虽然挑战赛的bbox只有一个
            # # 但我们还需要调用 merge_data_samples 对结果进行合并
            result_samples = merge_data_samples(predict_results)

            # 取出对应的关键点的预测结果, pred_instances.keypoints shape is (检测框数量，关键点数量，2)，我们就一个检测框，所以索引是0
            keypoints = result_samples.pred_instances.keypoints[0, :, :]

            keypoints_list = []
            line = []
            line.append(str(id))
            # line.append(size)
            for i in range(np.shape(keypoints)[0]):
                # 索引得到不同的关键点热图
                x0, y0 = keypoints[i, 0], keypoints[i, 1]
                gt_x,gt_y = gt[i*3],gt[i*3+1]
                keypoints_list.append([x0, y0])
                MRE = np.sqrt((x0-gt_x)**2+(y0-gt_y)**2) * spacing
                sdr2 = 1 if MRE <=2 else 0
                
                print("image_id:{} pid: {} predict ({},{}) gt ({},{})) MRE: {}".format(id,i,x0,y0,gt_x,gt_y,MRE))    
                line.append(sdr2)
                line.append(np.round(MRE,4))
                total_sdr_list.append(sdr2)
                total_mre_list.append(MRE)
                
            print(len(keypoints_list))
            fg_dataframe.loc[len(fg_dataframe)] = line


    mean_sdr = np.mean(total_sdr_list)
    mean_mre = np.mean(total_mre_list)
    std_mre = np.std(total_mre_list)
    print("mean_sdr: {} mean_mre: {} std_mre: {}".format(mean_sdr,mean_mre,std_mre))

    mean_row = fg_dataframe.mean()
    mean_row["image_id"] = "mean"


    fg_dataframe.loc[len(fg_dataframe)] = mean_row
    print(fg_dataframe)

    # 保存在和checkpoint 同级
    csv_path = os.path.join(os.path.dirname(config.load_weight_path), "fine_grained_result({}).csv".format(config.mode))
    fg_dataframe.to_csv(csv_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config file | 模型的配置文件
    parser.add_argument('--config_file', type=str, default='cldetection_configs_teststage/exp60_td-hm_hrnet-w32_udp-8xb64-250e-1024x1024_hm1024_KeypointMSELoss_sigma6_moreaug1_longer_dark_trainval_fold_outerbbox.py')

    # data parameters | 数据文件路径和配置文件的路径
    parser.add_argument('--load_mha_path', type=str, default='./step5_docker_and_upload/test/stack1.mha')
    parser.add_argument('--save_json_path', type=str, default='./step5_docker_and_upload/test/expected_output.json')

    # model load dir path | 最好模型的权重文件路径
    parser.add_argument('--load_weight_path', type=str, default='MMPose-checkpoints/exp60/best_SDR 2.0mm_epoch_28.pth')

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)
    # val or test
    parser.add_argument('--mode', type=str, default="test")

    experiment_config = parser.parse_args()
    main(experiment_config)
