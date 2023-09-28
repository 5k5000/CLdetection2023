# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import random
import argparse
import numpy as np
from PIL import Image

from cldetection_utils import load_train_stack_data, extract_one_image_landmarks, save_coco_json_dataset, remove_zero_padding, check_and_make_dir
from tqdm import tqdm

def main(config):
    # load image array and train ground truth landmarks json file
    train_stack_array = load_train_stack_data(config.mha_file_path)
    with open(config.train_gt_path, mode='r', encoding='utf-8') as f:
        train_gt_dict = json.load(f)

    # remove the redundant 0 padding, this operation not affects the processing of the location of the points
    check_and_make_dir(config.image_save_dir)
    all_image_shape_list = []
    for i in tqdm(range(np.shape(train_stack_array)[0])):
        # break
        image_array = train_stack_array[i, :, :, :]
        image_array = remove_zero_padding(image_array)
        all_image_shape_list.append(np.shape(image_array))
        pillow_image = Image.fromarray(image_array)
        pillow_image.save(os.path.join(config.image_save_dir, '%s.png' % (i + 1)))

    # parse out the landmark annotations corresponding to each image（解析出来每个图像对应的关键点标注）
    all_image_landmarks_list = []
    for i in tqdm(range(400)):
        image_landmarks = extract_one_image_landmarks(all_gt_dict=train_gt_dict, image_id=i + 1)
        all_image_landmarks_list.append(image_landmarks)

    # shuffle the order of the images list（打乱图像列表的顺序）
    random.seed(2023)
    all_index_list = [i for i in range(len(all_image_shape_list))]
    random.shuffle(all_index_list)

    # split the training set, validation set and test set, and save as json coco files（划分训练集，验证集和测试集，COCO格式）
    train_json_path = os.path.join(os.path.dirname(config.image_save_dir), 'train.json')
    print('Train JSON Path:', train_json_path)
    save_coco_json_dataset(image_shape_list=[all_image_shape_list[i] for i in all_index_list[0:300]],
                           image_landmarks_list=[all_image_landmarks_list[i] for i in all_index_list[0:300]],
                           image_ids_list=[i + 1 for i in all_index_list[0:300]],
                           json_save_path=train_json_path)

    valid_json_path = os.path.join(os.path.dirname(config.image_save_dir), 'valid.json')
    print('Valid JSON Path:', valid_json_path)
    save_coco_json_dataset(image_shape_list=[all_image_shape_list[i] for i in all_index_list[300:350]],
                           image_landmarks_list=[all_image_landmarks_list[i] for i in all_index_list[300:350]],
                           image_ids_list=[i + 1 for i in all_index_list[300:350]],
                           json_save_path=valid_json_path)

    test_json_path = os.path.join(os.path.dirname(config.image_save_dir), 'test.json')
    print('Test JSON Path:', test_json_path)
    save_coco_json_dataset(image_shape_list=[all_image_shape_list[i] for i in all_index_list[350:400]],
                           image_landmarks_list=[all_image_landmarks_list[i] for i in all_index_list[350:400]],
                           image_ids_list=[i + 1 for i in all_index_list[350:400]],
                           json_save_path=test_json_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--mha_file_path', type=str, default='./data/challenges/CLdetection2023/train_stack.mha')
    parser.add_argument('--train_gt_path', type=str, default='./data/challenges/CLdetection2023/train-gt.json')

    # save processed images dir path
    parser.add_argument('--image_save_dir', type=str, default='./data/challenges/CLdetection2023/preprocessed')

    experiment_config = parser.parse_args()
    main(experiment_config)
