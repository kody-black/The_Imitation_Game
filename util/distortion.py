# coding=utf-8
import numpy as np


def random_distortion(resimg, char_surf_ls, surf_augmentor):
    """
    对原图像和各字符图像做相同的random_distortion
    :param resimg: 原图像
    :param char_surf_ls: 各字符图像
    :param surf_augmentor: 数据增强
    :return: 扭曲后图像和各字符图像
    """
    all_images = [resimg] + char_surf_ls

    surf_augmentor.augmentor_images = [all_images]
    all_distorted_images = surf_augmentor.sample(1)[0]

    resimg = all_distorted_images[0]
    return resimg, all_distorted_images[1:]


def get_bbs(all_distorted_images):
    bbs = []
    for char_distorted_surf in all_distorted_images:
        coordinate = np.where(char_distorted_surf != 0)
        left, right = np.min(coordinate[1]), np.max(coordinate[1])
        top, bottom = np.min(coordinate[0]), np.max(coordinate[0])
        bbs.append([left, top, right, bottom])
    return bbs
