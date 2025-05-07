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
    根据给定的文本、字体和配置，在图像上绘制文本。文本可以根据配置进行旋转、倾斜等效果的处理。
    
    :param text: 要绘制的文本字符串。
    :param font: pygame.font.Font对象，定义了文本的字体和大小。
    :param cfg: 一个字典，包含了文本绘制时的配置信息，如曲线程度、旋转程度、字符的垂直和水平偏移等。
    :return: 返回一个包含绘制好的图像数组、字符边界框列表和字符边界轮廓列表的元组。
    """
    # 随机选择一个曲线中心点
    curve_center = random.randint(0, len(text) - 1)
    # 获取行间距
    line_spacing = font.get_sized_height() + 1
    # 获取文本边界
    line_bounds = font.get_rect(text)
    print("line_bounds: ", line_bounds)
    # 设定图像尺寸
    fsize = (round(5.0 * line_bounds.width), round(2 * line_spacing))

    # 曲线和旋转处理
    mid_idx = curve_center
    # 根据配置随机生成曲线程度
    curve_rate = random.random() * (cfg['curve_rate_max'] - cfg['curve_rate_min']) + cfg['curve_rate_min']
    # 根据配置随机正态分布生成曲线程度
    # curve_rate = np.random.normal((cfg['curve_rate_max'] + cfg['curve_rate_min']) / 2, (cfg['curve_rate_max'] - cfg['curve_rate_min']) / 6)

    # 根据配置随机生成旋转程度
    rotation_rate = random.random() * (cfg['rotation_rate_max'] - cfg['rotation_rate_min']) + cfg['rotation_rate_min']
    # 计算每个字符的曲线偏移量
    curve = [curve_rate * (i - mid_idx) * (i - mid_idx) for i in range(len(text))]
    print("curve: ", curve)
    # 计算每个字符的旋转角度
    rots = [-int(math.degrees(math.atan(2 * rotation_rate * (i - mid_idx) / (font.size / 2)))) for i in range(len(text))]

    bbs = [] # 字符边界框列表
    contours_ls = [] # 字符边界轮廓列表
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32) # 创建一个新的透明表面用于绘制文字
    rect = font.get_rect(text[0]) # 获取第一个字符的边界框
    rect.x = 20 # 设置起始x坐标
    rect.y = surf.get_rect().centery - np.sum(curve[:mid_idx]) # 设置起始y坐标，考虑曲线偏移
    shifted_rect = copy.deepcopy(rect) # 复制一份rect用于修改
    # 对第一个字符进行随机曲线偏移和旋转
    shifted_rect.y += random.randint(cfg["random_curve_min"], cfg["random_curve_max"])
    rot = rots[0] + random.randint(cfg["random_rotation_min"], cfg["random_rotation_max"])

    # 在表面上绘制旋转后的字符，并获取字符的边界框
    ch_bounds = font.render_to(surf, shifted_rect, text[0], rotation=rot)
    alpha_arr = pygame.surfarray.pixels_alpha(surf) # 获取表面的alpha通道数组
    last_rect = rect # 保存当前字符的rect，用于下一个字符的定位

    # 使用OpenCV找到字符的边界轮廓
    contours, hierarchy = cv2.findContours(alpha_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.concatenate(contours, axis=0) # 合并轮廓点
    bbs.append(ch_bounds) # 保存字符边界框
    contours_ls.append(contours) # 保存字符边界轮廓

    for i in range(1, len(text)):
        ch = text[i]  # 获取当前字符
        new_rect = font.get_rect(ch)  # 获取当前字符的边界框
        new_rect.y = last_rect.y  # 设置当前字符的起始y坐标与上一个字符一致
        # 根据曲线位置调整字符的y坐标
        new_rect.topleft = (last_rect.topright[0] + cfg["random_curve_max"] + 10, new_rect.topleft[1])
        if i <= mid_idx:
            new_rect.centery = new_rect.centery + curve[i - 1]
        else:
            new_rect.centery = new_rect.centery - curve[i]

        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)  # 为当前字符创建一个新的透明表面
        shifted_rect = copy.deepcopy(new_rect)  # 复制当前字符的rect用于修改
        # 对当前字符进行随机曲线偏移和旋转
        shifted_rect.y += random.randint(cfg["random_curve_min"], cfg["random_curve_max"])
        rot = rots[i] + random.randint(cfg["random_rotation_min"], cfg["random_rotation_max"])
        
        # 根据cfg['font_strong_plus_prob']的值随机决定是否要加粗该字符
        # if np.random.rand() < cfg['font_strong_plus_prob']:
        #     # 如果决定加粗，将字符绘制多次，每次稍微偏移一点
        #     for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        #         shifted_rect.x += dx
        #         shifted_rect.y += dy
        #         font.render_to(surf, shifted_rect, ch, rotation=rot)
        # else:
        #     # 否则，只绘制一次字符
        #     font.render_to(surf, shifted_rect, ch, rotation=rot)
        
        bbrect = font.render_to(surf, shifted_rect, ch, rotation=rot)  # 在表面上绘制旋转后的字符，并获取边界框

        alpha_surf = pygame.surfarray.pixels_alpha(surf)  # 获取当前表面的alpha通道数组

        # 对当前字符进行随机位移处理，以模拟字符间的距离
        dis = random.randint(cfg["random_dis_min"], cfg["random_dis_max"])
        while True:
            roll_alpha_surf = np.roll(alpha_surf, -1, axis=0)  # 水平位移alpha数组
            mask_1 = alpha_arr.astype(np.int64)
            mask_2 = roll_alpha_surf.astype(np.int64)
            # 检查位移后是否有重叠
            if np.sum((mask_1 + mask_2) == 510) == 0:
                alpha_surf = roll_alpha_surf
                bbrect.x -= 1
            else:
                alpha_surf = np.roll(alpha_surf, dis, axis=0)  # 位移alpha数组
                break

        # 合并当前字符的alpha通道数组到整体图像的alpha通道数组
        alpha_arr = np.clip(alpha_arr.astype(np.int64) + alpha_surf.astype(np.int64), 0, 255).astype(np.uint8)
        last_rect = new_rect  # 更新last_rect为当前字符的rect

        # 使用OpenCV找到当前字符的边界轮廓
        contours, hierarchy = cv2.findContours(alpha_surf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours, axis=0)  # 合并轮廓点
        bbs.append(bbrect)  # 保存字符边界框
        contours_ls.append(contours)  # 保存字符边界轮廓

    return alpha_arr, bbs, contours_ls


def cut_text(alpha_arr, bbs, contours_ls):
    """
    根据bbs方框，切割出文字区域，并修改边界的坐标点
    :param alpha_arr: 图像
    :param bbs: 边界框
    :param contours_ls: 边界坐标
    :return: 文本区域和坐标点
    """
    # 初始处理，根据第一个字符的边界框创建一个pygame.Rect对象
    r0 = pygame.Rect(bbs[0])
    # 计算所有字符边界框的并集，得到包含整个文本的最小矩形区域
    rect_union = r0.unionall(bbs)
    # 根据得到的并集矩形，裁剪alpha通道数组
    alpha_arr = alpha_arr[rect_union[0]:rect_union[0] + rect_union[2], rect_union[1]:rect_union[1] + rect_union[3]]
    for rect in contours_ls:
        rect[:, 0, 1] -= rect_union[0]
        rect[:, 0, 0] -= rect_union[1]

    return alpha_arr, contours_ls
