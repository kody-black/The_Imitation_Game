# coding=utf-8
import random
import math
import pygame
from pygame import freetype
import pygame.locals
import cv2
import numpy as np
import copy
import yaml


def draw_text(text: str, font: freetype.Font, cfg: dict) -> tuple[np.ndarray, list, list]:
    """
    根据给定的文本、字体和配置，在图像上绘制文本。文本可以根据配置进行旋转、倾斜等效果的处理。

    :param text: 要绘制的文本字符串。
    :param font: freetype.Font对象，定义了文本的字体和大小。
    :param cfg: 一个字典，包含了文本绘制时的配置信息，如曲线程度、旋转程度、字符的垂直和水平偏移等。
    :return: 返回一个包含绘制好的图像数组、字符边界框列表和字符边界轮廓列表的元组。
    """

    # 设置字符方向曲线
    curve_center = random.randint(0, len(text) - 1)
    curve_rate = (
        random.random() * (cfg["curve_rate_max"] - cfg["curve_rate_min"])
        + cfg["curve_rate_min"]
    )
    curve = [
        curve_rate * (i - curve_center) * (i - curve_center) for i in range(len(text))
    ]

    rotation_rate = (
        random.random() * (cfg["rotation_rate_max"] - cfg["rotation_rate_min"])
        + cfg["rotation_rate_min"]
    )
    rotation_rate = int(rotation_rate)
    rots = [rotation_rate * (i - curve_center) for i in range(len(text))]

    # 单个字符的随机配置
    rots = [
        rots[i] + random.randint(cfg["random_rotation_min"], cfg["random_rotation_max"])
        for i in range(len(text))
    ]
    strong = [random.random() < cfg["font_strong_prob"] for i in range(len(text))]
    underline = [random.random() < cfg["font_underline_prob"] for i in range(len(text))]
    oblique = [random.random() < cfg["font_oblique_prob"] for i in range(len(text))]
    spacing_y = [
        random.randint(cfg["random_curve_min"], cfg["random_curve_max"])
        for i in range(len(text))
    ]
    # 随机字符垂直间距正态分布random_curve_min到random_curve_max
    # spacing_y = [np.random.normal((cfg["random_curve_max"] + cfg["random_curve_min"]) / 2, (cfg["random_curve_max"] - cfg["random_curve_min"]) / 6) for i in range(len(text))]
    spacing_x = [
        random.randint(cfg["random_dis_min"], cfg["random_dis_max"])
        for i in range(len(text))
    ]
    # 随机字符水平间距正态分布random_dis_min到random_dis_max
    # spacing_x = [np.random.normal((cfg["random_dis_max"] + cfg["random_dis_min"]) / 2, (cfg["random_dis_max"] - cfg["random_dis_min"]) / 6) for i in range(len(text))]

    bbs = []
    contours_ls = []
    bbs_color = (0, 255, 0)

    # 保证画布足够大
    width = sum([font.get_rect(char).width for char in text]) * 5
    height = font.get_rect(text).height * 20

    now_x = width // 5  # 设置起始x坐标
    now_y = height // 2  # 设置起始y坐标

    for i, char in enumerate(text):
        surf = pygame.Surface(
            (width, height), pygame.locals.SRCALPHA, 32
        )  # 包含一个 alpha 通道的图层
        font.strong = strong[i]
        font.underline = underline[i]
        font.oblique = oblique[i]
        rotation = rots[i]
        rect = font.render_to(surf, (now_x, now_y), char, rotation=rotation)

        # 如果决定进一步加粗，将字符绘制多次，每次稍微偏移一点
        strong_value = cfg["font_strong_value"]
        if font.strong and np.random.rand() < cfg["font_strong_plus_prob"]:
            shift_x = -strong_value
            while shift_x <= strong_value:
                shift_y = -strong_value
                while shift_y <= strong_value:
                    font.render_to(
                        surf,
                        (now_x + shift_x, now_y + shift_y),
                        char,
                        rotation=rotation,
                    )
                    shift_y += 1
                shift_x += 1
            bbs.append(
                [
                    now_x - strong_value,
                    now_y - strong_value,
                    rect.width + 2 * strong_value,
                    rect.height + 2 * strong_value,
                ]
            )
        else:
            bbs.append([now_x, now_y, rect.width, rect.height])

        now_x += rect.width + spacing_x.pop(0)
        if i <= curve_center:
            now_y += curve[i]
        else:
            now_y -= curve[i]
        now_y += spacing_y.pop(0)

        alpha_surf = pygame.surfarray.pixels_alpha(surf)
        if i == 0:
            alpha_arr = alpha_surf
        else:
            alpha_arr = np.clip(
                alpha_arr.astype(np.int64) + alpha_surf.astype(np.int64), 0, 255
            ).astype(np.uint8)

        contours, _ = cv2.findContours(
            alpha_surf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = np.concatenate(contours, axis=0)
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
    # 初始处理，根据第一个字符的边界框创建一个pygame.Rect对象
    r0 = pygame.Rect(bbs[0])
    # 计算所有字符边界框的并集，得到包含整个文本的最小矩形区域
    rect_union = r0.unionall(bbs)
    # 根据得到的并集矩形，裁剪alpha通道数组
    alpha_arr = alpha_arr[
        rect_union[0] : rect_union[0] + rect_union[2],
        rect_union[1] : rect_union[1] + rect_union[3],
    ]
    for rect in contours_ls:
        rect[:, 0, 1] -= rect_union[0]
        rect[:, 0, 0] -= rect_union[1]

    return alpha_arr, contours_ls


if __name__ == "__main__":
    from font import get_font

    with open(f"../cfg/cfg_google.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg)

    font_name = "../font/" + cfg["font_name"]
    text = "abingleda"

    freetype.init()
    font = get_font(font_name, cfg)
    alpha_arr, bbs, contours_ls = draw_text(text, font, cfg)
    # 保存图片
    cv2.imwrite("text.png", alpha_arr.swapaxes(0, 1))
    # 剪切图片
    alpha_arr, contours_ls = cut_text(alpha_arr, bbs, contours_ls)
    # 保存图片
    cv2.imwrite("text_cut.png", alpha_arr.swapaxes(0, 1))