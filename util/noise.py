import numpy as np
from PIL.ImageDraw import Draw
from PIL import Image
import random


def create_dots(img):
    image = Image.fromarray(img)
    draw = Draw(image)
    w, h = image.size
    width = random.randint(1, 5)
    number = 50
    for _ in range(number):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 + random.randint(-2, 2), y1 + random.randint(-2, 2))), fill=255, width=width)
    image = np.array(image)
    return image


def create_lines(img, num=1, thickness=1):
    image = Image.fromarray(img)
    draw = Draw(image)
    w, h = image.size
    width = thickness
    number = num
    for _ in range(number):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        draw.line(((x1, y1), (x2, y2)), fill=255, width=width)
    image = np.array(image)
    return image



def create_arcs(img, thickness=1, num=1):
    """
    在图像上绘制正弦曲线干扰线（优化版）。
    使用Numpy向量化操作替代Python循环，以提高性能。
    """
    height, width = img.shape[:2]
    if width == 0 or height == 0: return img
    
    for _ in range(num):
        # 生成 x 坐标
        x_coords = np.arange(width)
        
        # 正弦波的随机参数
        amplitude = random.randint(0, height)
        # 避免当宽度很小时除以零
        frequency = random.randint(max(1, width // 10), width)
        phase = random.uniform(0, 2 * np.pi)
        
        # 计算基础 y 坐标
        y_base = (np.sin(x_coords / frequency * 2 * np.pi + phase) + 1) * (amplitude / 2)

        # 通过偏移基础y坐标来实现曲线的厚度
        for offset in range(-thickness // 2, thickness // 2 + 1):
            y_coords = (y_base + offset).astype(int)
            
            # 创建一个布尔掩码以选择在图像边界内的坐标
            valid_mask = (y_coords >= 0) & (y_coords < height)
            
            # 使用掩码同时索引 x 和 y 坐标，并设置像素值
            img[y_coords[valid_mask], x_coords[valid_mask]] = 255
            
    return img


def create_polynomial(img, thickness=2, num=1):
    """
    在图像上绘制多项式曲线干扰线（优化版）。
    使用Numpy向量化操作替代Python循环，以提高性能。
    """
    height, width = img.shape[:2]
    if width == 0 or height == 0: return img

    for _ in range(num):
        # 定义多项式的阶数
        degree = random.randint(2, 5)
        # 为多项式方程生成随机系数
        coeffs = np.random.uniform(-1, 1, degree + 1)
        # 定义一个随机的垂直偏移
        vertical_offset = random.uniform(-0.5 * height, 0.5 * height)

        # 生成归一化的 x 值 [0, 1] 以获得更好的数值稳定性
        x = np.linspace(0, 1, num=width)
        
        # 使用向量化多项式求值计算 y 值
        y_base = np.polynomial.polynomial.polyval(x, coeffs)
        # 缩放并偏移曲线以适应图像
        y_base = height / 2 + y_base * height / 4 + vertical_offset

        x_coords = np.arange(width)

        # 通过偏移基础y坐标来实现曲线的厚度
        for offset in range(-thickness // 2, thickness // 2 + 1):
            y_coords = (y_base + offset).astype(int)
            
            # 创建一个布尔掩码以选择在图像边界内的坐标
            valid_mask = (y_coords >= 0) & (y_coords < height)
            
            # 使用掩码同时索引 x 和 y 坐标，并设置像素值
            img[y_coords[valid_mask], x_coords[valid_mask]] = 255
            
    return img
