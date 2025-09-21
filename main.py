# coding=utf-8
import pickle
from pygame import freetype
import cv2
from PIL import Image
import numpy as np
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


def process_args():
    # 使用时传入参数 --dataset dataset_name，默认为google
    # 对应的config文件为cfg/cfg_google.yaml
    parser = argparse.ArgumentParser(description="Captcha Generator")
    parser.add_argument("--dataset", default="google", type=str, help="Dataset_name")
    args = parser.parse_args()
    db = args.dataset
    try:
        with open(f"cfg/cfg_{db}.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            print(f"Load configuration from cfg/cfg_{db}.yaml")
    except FileNotFoundError:
        print(f"Failed to load configuration from cfg/cfg_{db}.yaml, please check the file.")
        exit(1)
    return cfg, db

def main():
    cfg, db = process_args()
    freetype.init()

    # 如果有all_chars配置项，则使用它，否则使用char_type
    if "all_chars" in cfg and len(cfg["all_chars"]) > 0:
        chars = cfg["all_chars"]
    else:
        chars = CHAR_TYPES[cfg["char_type"]]
        # 剔除掉ex_char
        if "ex_char" in cfg:
            for ex_char in cfg["ex_char"]:
                chars = chars.replace(ex_char, "")

    font_name = "font/" + cfg["font_name"]

    # --- 使用while循环和set来确保生成足量的唯一验证码 ---
    target_count = cfg["captcha_number"]
    generated_texts = set()
    generated_count = 0

    # 确保输出目录存在
    img_output_dir = f"output/image/{db}"
    obj_output_dir = f"output/object/{db}"
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(obj_output_dir, exist_ok=True)
    
    print(f"Attempting to generate {target_count} unique captchas...")

    while generated_count < target_count:
        text = "".join(random.choices(chars, k=random.randint(cfg["min_len"], cfg["max_len"])))
        
        # 如果文本已存在，则跳过此次循环
        if text in generated_texts:
            continue
            
        font = get_font(font_name, cfg)
        
        alpha_arr, bbs, char_surf_ls = draw_text(text, font, cfg)
        alpha_arr, char_surf_ls = cut_text(alpha_arr, bbs, char_surf_ls)
        resimg = alpha_arr.swapaxes(0, 1)

        resimg, char_surf_ls = perspective(resimg, char_surf_ls, cfg)

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

        bbs = get_bbs(char_surf_ls)

        if cfg["is_bg"]:
            img_bg_path = cfg["bg_dir"]
            l_out = background_colorize(resimg,cfg["font_color"],img_bg_path)
        else:
            l_out = pure_colorize(resimg, cfg["font_color"], cfg["background_color"])
        
        # --- 智能缩放和填充到目标尺寸 ---
        current_h, current_w = l_out.shape[:2]
        target_w, target_h = cfg["image_width"], cfg["image_height"]

        scale = min(target_w / current_w, target_h / current_h)
        resized_w, resized_h = int(current_w * scale), int(current_h * scale)

        img_pil = Image.fromarray(cv2.cvtColor(l_out, cv2.COLOR_BGR2RGB))
        resized_img_pil = img_pil.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        # --- 修正拼写错误: COLOR_RGB_BGR -> COLOR_RGB2BGR ---
        resized_img_cv = cv2.cvtColor(np.array(resized_img_pil), cv2.COLOR_RGB2BGR)

        if len(cfg["background_color"]) > 0:
            bg_color_bgr = cfg["background_color"][0][::-1]
        else:
            bg_color_bgr = (255, 255, 255)
        final_canvas = np.full((target_h, target_w, 3), bg_color_bgr, dtype=np.uint8)

        paste_x = (target_w - resized_w) // 2
        paste_y = (target_h - resized_h) // 2

        final_canvas[paste_y:paste_y+resized_h, paste_x:paste_x+resized_w] = resized_img_cv
        
        l_out = final_canvas

        updated_bbs = []
        for left, top, right, bottom in bbs:
            new_left = int(left * scale) + paste_x
            new_top = int(top * scale) + paste_y
            new_right = int(right * scale) + paste_x
            new_bottom = int(bottom * scale) + paste_y
            updated_bbs.append([new_left, new_top, new_right, new_bottom])
        bbs = updated_bbs

        # # --- [调试代码] 恢复: 在图中绘制出bbs边框和轮廓 ---
        # # 需要时取消下面的注释来可视化边界框和轮廓
        # # 绘制bbs边框 (绿色)
        # for left, top, right, bottom in bbs:
        #     cv2.rectangle(l_out, (left, top), (right, bottom), (0, 255, 0), 1)
        
        # # 绘制字符轮廓 (红色)
        # # 注意：此操作会重新计算并缩放轮廓，可能略微降低生成速度
        # for char_surf in char_surf_ls:
        #     # 找到轮廓点坐标 (y, x)
        #     coords = np.argwhere(char_surf > 0)
        #     if coords.size == 0:
        #         continue
        
        #     # 应用与图像相同的缩放和平移
        #     scaled_coords = coords * scale
        #     translated_coords = scaled_coords + [paste_y, paste_x]
        #     final_coords = translated_coords.astype(int)
        
        #     # 在最终画布上绘制点
        #     for y, x in final_coords:
        #         if 0 <= y < target_h and 0 <= x < target_w:
        #             l_out[y, x] = [0, 0, 255] # BGR for red

        # --- 保存文件 ---
        if cfg["mode"] == "develop":
            # 在开发模式下，使用计数器作为文件名以避免覆盖
            filename = f"{generated_count}_{text}"
        else:
            filename = text

        cv2.imwrite(f"{img_output_dir}/{filename}.png", l_out)
        with open(f"{obj_output_dir}/{filename}.pkl", "wb") as f:
            pickle.dump(bbs, f)

        # --- 更新计数器和集合 ---
        generated_texts.add(text)
        generated_count += 1

        # 打印进度
        if generated_count % 100 == 0:
            print(f"  ... {generated_count} / {target_count} generated ...")

    print(f"\nSuccessfully generated {generated_count} unique captchas in output/image/{db}/")


if __name__ == "__main__":
    main()

