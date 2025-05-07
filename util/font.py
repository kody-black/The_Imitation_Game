# coding=utf-8
import random
from pygame import freetype
import numpy as np


def get_font(font, cfg):
    """
    根据配置设置并返回字体
    :param font_list: 字体
    :param cfg: 字体的配置
    :return: 字体
    """
    font = freetype.Font(font)
    font.antialiased = cfg["font_antialiased"]
    font.origin = cfg["font_origin"]
    font.size = random.randint(cfg["font_size_min"], cfg["font_size_max"])
    if np.random.rand() < cfg["font_underline_prob"]:
        font.underline = True
    else:
        font.underline = False
    if np.random.rand() < cfg["font_strong_prob"]:
        font.strong = True
    else:
        font.strong = False
    font.strong = True
    if np.random.rand() < cfg["font_oblique_prob"]:
        font.oblique = True
    else:
        font.oblique = False
    return font
