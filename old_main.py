# coding=utf-8
import pickle
from pygame import freetype
import cv2
import Augmentor
import string
import random
import os
import argparse
import yaml

from util.font import get_font
from util.draw import draw_text, cut_text
from util.perspective import perspective
from util.distortion import random_distortion, get_bbs
from util.color import *
from util.waving import waving
from util.noise import create_dots, create_lines, create_arcs
from config import CHAR_TYPES


def main():
    # 使用时传入参数 --dataset dataset_name，默认为google
    parser = argparse.ArgumentParser(description="Captcha Generator")
    parser.add_argument("--dataset", default="google", type=str, help="Dataset_name")
    args = parser.parse_args()

    with open(f"cfg/cfg_{args.dataset}.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(f"load cfg from cfg/cfg_{args.dataset}.yaml")

    freetype.init()

    chars = CHAR_TYPES[cfg["char_type"]]
    # 剔除掉ex_char
    if "ex_chars" in cfg:
        for ex_char in cfg["ex_chars"]:
            chars = chars.replace(ex_char, "")

    font_name = "font/" + cfg["font_name"]
    # font_name = cfg["font_name"]

    
    surf_augmentor = Augmentor.DataPipeline(None)
    
    # 扭曲
    if cfg["distort_p"] != 0:
        surf_augmentor.random_distortion(
            probability=cfg["distort_p"],
            grid_width=cfg["grid_size"],
            grid_height=cfg["grid_size"],
            magnitude=cfg["magnitude"],
        )

    for captcha_index in range(cfg["captcha_number"]):
        # text = "".join(random.choices(chars, k=random.randint(cfg["min_len"], cfg["max_len"])))
        text = "abingleda"
        font = get_font(font_name, cfg)

        alpha_arr, bbs, contours_ls = draw_text(text, font, cfg)
        alpha_arr, contours_ls = cut_text(alpha_arr, bbs, contours_ls)
        img = alpha_arr.swapaxes(0, 1)

        resimg, char_surf_ls = perspective(img, contours_ls, cfg)

        if cfg["is_waving"]:
            resimg, char_surf_ls = waving(
                resimg,
                char_surf_ls,
                len(text),
                cfg["waving_orientation"],
                cfg["waving_level"],
                cfg["waving_period"],
            )

        if cfg["is_dots"]:
            resimg = create_dots(resimg)
        if cfg["is_line"]:
            resimg = create_lines(resimg)
        if cfg["is_arc"]:
            resimg = create_arcs(resimg, cfg["arc_thickness"], cfg["arc_num"])

        resimg, char_surf_ls = random_distortion(resimg, char_surf_ls, surf_augmentor)

        bbs = get_bbs(char_surf_ls)

        # l_out = pure_colorize(resimg, cfg["font_color"], cfg["background_color"])
        # l_out = no_colorize(resimg)
        # l_out = gradient_colorize(resimg)
        bg_dir = cfg["bg_dir"]
        l_out = background_colorize(resimg,cfg["font_color"],bg_dir)

        # 利用Image中的resize方法修改图片大小为cfg["image_width"]和cfg["image_height"]
        # 现在修改图片大小的问题是没有改变bbs边框和轮廓的坐标
        # out = Image.fromarray(l_out).resize((cfg["image_width"], cfg["image_height"]))
        # l_out = np.array(out)

        # # 在图中绘制出bbs边框
        # for left, top, right, bottom in bbs:
        #     cv2.rectangle(l_out, (left, top), (right, bottom), (0, 255, 0), 1)

        # # 在图中绘制出字符的轮廓
        # for char_surf in char_surf_ls:
        #     for i in range(char_surf.shape[0]):
        #         for j in range(char_surf.shape[1]):
        #             if char_surf[i, j] > 0:
        #                 l_out[i, j][:3] = [255, 0, 0]

        if cfg["mode"] != "develop":
            captcha_index = text

        if not os.path.exists(f"output/image/{args.dataset}"):
            os.makedirs(f"output/image/{args.dataset}")
        cv2.imwrite(f"output/image/{args.dataset}/{captcha_index}.png", l_out)

        if not os.path.exists(f"output/object/{args.dataset}"):
            os.makedirs(f"output/object/{args.dataset}")
        with open(f"output/object/{args.dataset}/{captcha_index}.pkl", "wb") as f:
            pickle.dump(bbs, f)


if __name__ == "__main__":
    main()
