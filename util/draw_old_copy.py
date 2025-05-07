# coding=utf-8
import random
import math
import pygame
import pygame.locals
import cv2
import numpy as np
import copy


def draw_text(text, font, cfg):
    """
    根据文本、字体和配置（旋转、垂直偏移、水平偏移）绘制图像
    整体：curve程度、旋转程度
    单个：垂直偏移、水平偏移、旋转
    :param text: 文本
    :param font: 字体
    :param cfg: 配置
    :return: 绘制图像、字符方框、字符边界
    """
    curve_center = random.randint(0, len(text) - 1)
    line_spacing = font.get_sized_height() + 1
    line_bounds = font.get_rect(text)
    fsize = (round(5.0 * line_bounds.width), round(10 * line_spacing))

    mid_idx = curve_center
    curve_rate = random.random() * (cfg['curve_rate_max'] - cfg['curve_rate_min']) + cfg['curve_rate_min']
    rotation_rate = random.random() * (cfg['rotation_rate_max'] - cfg['rotation_rate_min']) + cfg['rotation_rate_min']
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(len(text))]
    rots = [-int(math.degrees(math.atan(2 * rotation_rate * (i - mid_idx) / (font.size / 2)))) for i in
            range(len(text))]

    bbs = []
    contours_ls = []
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    rect = font.get_rect(text[0])
    rect.x = 20
    rect.y = surf.get_rect().centery - np.sum(curve[:mid_idx])
    shifted_rect = copy.deepcopy(rect)
    shifted_rect.y += random.randint(cfg["random_curve_min"], cfg["random_curve_max"])
    rot = rots[0] + random.randint(cfg["random_rotation_min"], cfg["random_rotation_max"])
    ch_bounds = font.render_to(surf, shifted_rect, text[0], rotation=rot)
    alpha_arr = pygame.surfarray.pixels_alpha(surf)
    last_rect = rect

    contours, hierarchy = cv2.findContours(alpha_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.concatenate(contours, axis=0)
    bbs.append(ch_bounds)
    contours_ls.append(contours)

    for i in range(1, len(text)):
        ch = text[i]
        new_rect = font.get_rect(ch)
        new_rect.y = last_rect.y
        new_rect.topleft = (last_rect.topright[0] + cfg["random_curve_max"] + 10, new_rect.topleft[1])
        if i <= mid_idx:
            new_rect.centery = new_rect.centery + curve[i - 1]
        else:
            new_rect.centery = new_rect.centery - curve[i]
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        shifted_rect = copy.deepcopy(new_rect)
        shifted_rect.y += random.randint(cfg["random_curve_min"], cfg["random_curve_max"])
        rot = rots[i] + random.randint(cfg["random_rotation_min"], cfg["random_rotation_max"])
        bbrect = font.render_to(surf, shifted_rect, ch, rotation=rot)
        alpha_surf = pygame.surfarray.pixels_alpha(surf)

        dis = random.randint(cfg["random_dis_min"], cfg["random_dis_max"])
        while True:
            roll_alpha_surf = np.roll(alpha_surf, -1, axis=0)
            mask_1 = alpha_arr.astype(np.int64)
            mask_2 = roll_alpha_surf.astype(np.int64)
            if np.sum((mask_1 + mask_2) == 510) == 0:
                alpha_surf = roll_alpha_surf
                bbrect.x -= 1
            else:
                alpha_surf = np.roll(alpha_surf, dis, axis=0)
                break
        alpha_arr = np.clip(alpha_arr.astype(np.int64) + alpha_surf.astype(np.int64), 0, 255).astype(np.uint8)
        last_rect = new_rect

        contours, hierarchy = cv2.findContours(alpha_surf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours, axis=0)
        bbs.append(bbrect)
        contours_ls.append(contours)

    return alpha_arr, bbs, contours_ls


def cut_text(alpha_arr, bbs, contours_ls):
    """
    根据bbs方框，切割出文字区域，并修改边界的坐标点
    :param alpha_arr: 图像
    :param bbs: 边界框
    :param contours_ls: 边界坐标
    :return: 文本区域和坐标点
    """
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)
    alpha_arr = alpha_arr[rect_union[0]:rect_union[0] + rect_union[2], rect_union[1]:rect_union[1] + rect_union[3]]
    for rect in contours_ls:
        rect[:, 0, 1] -= rect_union[0]
        rect[:, 0, 0] -= rect_union[1]

    return alpha_arr, contours_ls
