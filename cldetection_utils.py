# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import shutil
import numpy as np
import SimpleITK as sitk


def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError('Error, the provided path (%s) is a file path, not a folder path.' % dir_path)
        shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


def load_train_stack_data(file_path: str) -> np.ndarray:
    """
    function to load train_stack.mha data file | 加载train_stack.mha数据文件的函数
    :param file_path: train_stack.mha filepath | 挑战赛提供的train_stack.mha文件路径
    :return: a 4-dim array containing 400 training set cephalometric images | 一个包含了400张训练集头影图像的四维的矩阵
    """
    sitk_stack_image = sitk.ReadImage(file_path)
    np_stack_array = sitk.GetArrayFromImage(sitk_stack_image)
    return np_stack_array


def remove_zero_padding(image_array: np.ndarray) -> np.ndarray:
    """
    function to remove zero padding in an image | 去除图像中的0填充函数
    :param image_array: one cephalometric image array, shape is (2400, 2880, 3) | 一张头影图像的矩阵，形状为(2400, 2880, 3)
    :return: image matrix after removing zero padding | 去除零填充部分的图像矩阵
    """
    row = np.sum(image_array, axis=(1, 2))
    column = np.sum(image_array, axis=(0, 2))

    non_zero_row_indices = np.argwhere(row != 0)
    non_zero_column_indices = np.argwhere(column != 0)

    last_row = int(non_zero_row_indices[-1])
    last_column = int(non_zero_column_indices[-1])

    image_array = image_array[:last_row+1, :last_column+1, :]
    return image_array


def extract_one_image_landmarks(all_gt_dict: dict, image_id: int) -> dict:
    """
    function to extract landmark information corresponding to an image | 提出出对应图像的关键点信息
    :param all_gt_dict: a dict loaded from the train_gt.json file | 从train_gt.json文件加载得到的字典
    :param image_id: image id between 0 and 400 | 图像的id，在0到400之间
    :return: a dict containing pixel spacing and coordinates of 38 landmarks | 一个字典，包含了像素的spacing和38个关键点的坐标
    """
    image_dict = {'image_id': image_id}
    for landmark in all_gt_dict['points']:
        point = landmark['point']
        if point[-1] != image_id:
            continue
        image_dict['scale'] = float(landmark['scale'])
        image_dict['landmark_%s' % landmark['name']] = point[:2]
    return image_dict


def save_coco_json_dataset(image_shape_list: list, image_landmarks_list: list, image_ids_list: list, json_save_path: str,num_of_points=38) -> None:
    """
    function to save coco json dataset | 保存为COCO数据集
    :param image_shape_list: images shape list | 所有图像的shape大小列表
    :param image_landmarks_list: images landmarks list | 对应的图像的关键点列表
    :param image_ids_list: images id list | 对应的图像的ID列表
    :param json_save_path: json file save path | json文件的保存路径
    :return: None
    """
    json_dict = {}

    # header information
    json_dict['info'] = {'description': 'This is 1.0 version of the CL-Detection2023 MS COCO dataset.',
                         'url': 'https://cl-detection2023.grand-challenge.org',
                         'version': '1.0',
                         'year': 2023,
                         'contributor': 'CL-Detection Challenge Team'}

    # landmarks information
    for image_shape, image_landmarks, image_id in zip(image_shape_list, image_landmarks_list, image_ids_list):
        # image information
        image = {'id': image_id,
                 'file_name': '%s.png' % image_id,
                 'width': image_shape[1],
                 'height': image_shape[0],
                 'spacing': image_landmarks['scale']  # spacing is used for metrics calculation
                 }
        images = json_dict.get('images', [])
        images.append(image)
        json_dict['images'] = images

        # annotation information and bbox range
        keypoints = []
        for landmark_id in range(1, num_of_points+1):
            x, y = image_landmarks['landmark_%s' % landmark_id]
            keypoints.extend([x, y, 1])

        annotation = {'id': image_id,
                      'image_id': image_id,
                      'category_id': 1,
                      'keypoints': keypoints,
                      'num_keypoints': num_of_points,
                      'iscrowd': 0,
                      'bbox': [0, 0, image['width'], image['height']],
                      'area': image['width'] * image['height'],
                      }
        annotations = json_dict.get('annotations', [])
        annotations.append(annotation)
        json_dict['annotations'] = annotations

    # category information
    category = {'id': 1,
                'name': 'cephalometric'}
    json_dict['categories'] = [category]

    # save json dict as json file
    json_string = json.dumps(json_dict, indent=4)
    with open(json_save_path, mode='w', encoding='utf-8') as f:
        f.write(json_string)



