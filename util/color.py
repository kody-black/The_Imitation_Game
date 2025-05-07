# coding=utf-8
import os
import numpy as np
import scipy
import scipy.fftpack
import cv2
from scipy.spatial.distance import cdist
import random
from PIL import Image


def colorize(resimg, bg_augmentor, bg_list, colorsLAB, colorsRGB):
    # 从背景列表中随机选择一个背景图像并读取
    bg = cv2.imread(random.choice(bg_list))
    # 获取背景图像的高度和宽度
    bg_h, bg_w = bg.shape[:2]
    # 获取前景图像的高度和宽度
    surf_h, surf_w = resimg.shape[:2]
    # 随机计算前景图像在背景中的位置
    x = np.random.randint(0, bg_w - surf_w + 1)
    y = np.random.randint(0, bg_h - surf_h + 1)
    # 提取背景中对应位置的区域
    t_b = bg[y:y + surf_h, x:x + surf_w, :]

    # 准备背景增强器的输入格式
    bgs = [[t_b]]
    # 将提取的背景区域赋值给背景增强器
    bg_augmentor.augmentor_images = bgs
    # 对背景区域进行增强处理
    t_b = bg_augmentor.sample(1)[0][0]

    # 将增强后的背景从RGB色彩空间转换到Lab色彩空间
    bg_mat = cv2.cvtColor(t_b, cv2.COLOR_RGB2Lab)
    # 将背景图像重塑为二维数组，用于颜色匹配
    bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]), 3))
    # 计算背景和指定颜色之间的欧式距离
    norms = cdist(bg_mat, colorsLAB, metric="euclidean")
    # 找到与背景颜色最接近的颜色索引
    nn = np.argmin(norms, axis=1)
    # 对颜色进行调整，这里是一种简单的排列
    colorsRGB = np.r_[colorsRGB[:, 6:12], colorsRGB[:, 0:6]].astype(np.uint8)
    # 根据索引选择对应的颜色
    data_col = colorsRGB[nn, :]

    # 对选择的颜色进行随机调整以增加多样性
    col_sample = data_col[:, :3] + data_col[:, 3:6] * np.random.randn(data_col.shape[0], 1)
    # 确保调整后的颜色值在有效范围内
    col_sample = np.clip(col_sample, 0, 255).astype(np.uint8)
    # 将调整后的颜色重新塑形，形成与前景图像相匹配的颜色图
    fg_col = np.reshape(col_sample, (t_b.shape[0], t_b.shape[1], 3))

    # 前景的alpha图层（透明度）
    fg_alpha = resimg
    # 将前景颜色应用到整个前景图像上
    fg_color = np.ones((fg_alpha.shape[0], fg_alpha.shape[1], 3), dtype=np.uint8) * fg_col

    # 计算背景颜色的平均值，用于背景颜色填充
    bg_col = np.mean(np.mean(t_b, axis=0), axis=0)
    # 创建背景的alpha图层
    bg_alpha = 255 * np.ones_like(fg_alpha, dtype=np.uint8)
    # 将背景颜色应用到整个背景图像上
    bg_color = np.ones((bg_alpha.shape[0], bg_alpha.shape[1], 3), dtype=np.uint8) * bg_col[None, None, :]

    # 计算前景和背景的透明度混合
    a_f = fg_alpha / 255.0
    a_b = bg_alpha / 255.0
    # 最终的前景和背景颜色
    c_f = fg_color
    c_b = bg_color

    # 计算最终图像的颜色值
    c_r = ((1 - a_f) * a_b)[:, :, None] * c_b + a_f[:, :, None] * c_f

    # 将计算结果转换为uint8格式
    r_color = c_r.astype(np.uint8)

    # 使用泊松融合将前景色和背景色融合到一起
    l_out = poisson_blit_images(r_color.copy(), t_b.copy())
    return l_out


def pure_colorize(fg_alpha, fg_cols, bg_cols):
    fg_col = np.array(fg_cols[np.random.randint(0, len(fg_cols))], dtype=np.uint8)
    fg_color = np.ones((fg_alpha.shape[0], fg_alpha.shape[1], 3), dtype=np.uint8) * fg_col[None, None, :]

    bg_col = np.array(bg_cols[np.random.randint(0, len(bg_cols))], dtype=np.uint8)
    bg_color = np.ones((fg_alpha.shape[0], fg_alpha.shape[1], 3), dtype=np.uint8) * bg_col[None, None, :]
    
    a_f = fg_alpha / 255.0
    c_f = fg_color
    c_b = bg_color

    # Alpha Blending 公式
    c_r = a_f[:, :, None] * c_f + (1 - a_f[:, :, None]) * c_b

    # 将计算结果转换为uint8格式
    l_out = c_r.astype(np.uint8)
    return l_out

def pure_colorize2(fg_alpha, fg_cols, bg_cols):
    # 利用image进行Image.fromarray(fg_alpha)
    fg_img = Image.fromarray(fg_alpha)
    # 从fg_cols中随机选择一个前景颜色
    fg_col = fg_cols[np.random.randint(0, len(fg_cols))]
    # 将前景图像的颜色修改为fg_col
    fg_img = fg_img.convert("RGBA")
    data = fg_img.getdata()
    new_data = []
    for item in data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(fg_col)
    fg_img.putdata(new_data)
    # 将前景图像转换为numpy数组
    l_out = np.array(fg_img)
    return l_out

# 将文字设置为随机的渐变色
def gradient_colorize(resimg):
    # 生成两个随机颜色作为渐变的起止颜色
    start_col = (np.random.rand(3) * 255.).astype(np.uint8)
    end_col = (np.random.rand(3) * 255.).astype(np.uint8)

    # 获取图像的高度和宽度
    height, width = resimg.shape[:2]

    # 创建一个宽度方向的线性渐变
    gradient = np.ones((height, width, 3), dtype=np.float32)
    for i in range(3):  # 对于RGB的每一层
        gradient[:, :, i] = np.linspace(start_col[i], end_col[i], width)

    # 获取前景图像的alpha层
    fg_alpha = resimg

    # 计算前景的alpha值（透明度值），转换为0到1之间的小数
    a_f = fg_alpha / 255.0

    # 设置背景颜色为纯白色
    bg_col = np.array((255, 255, 255), dtype=np.uint8)
    bg_alpha = 255 * np.ones_like(fg_alpha, dtype=np.uint8)
    bg_color = np.ones((bg_alpha.shape[0], bg_alpha.shape[1], 3), dtype=np.uint8) * bg_col[None, None, :]

    # 计算前景和背景的alpha值
    a_b = bg_alpha / 255.0

    # 将渐变颜色应用于前景色
    c_f = gradient

    # 将背景颜色赋值给变量c_b
    c_b = bg_color

    # 计算最终图像的颜色值
    c_r = ((1 - a_f) * a_b)[:, :, None] * c_b + a_f[:, :, None] * c_f
   
    # 将计算结果转换为uint8格式
    l_out = c_r.astype(np.uint8)

    return l_out

def background_colorize(resimg, forecolors, bg_dir):
    forecolor = forecolors[np.random.randint(0, len(forecolors))]
    fg_img = Image.fromarray(resimg)

    # 从bg_dir中随机选择一个背景图片
    bg_files = [os.path.join(bg_dir, file) for file in os.listdir(bg_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    if not bg_files:
        raise FileNotFoundError("No background images found in the specified directory.")
    bg_path = random.choice(bg_files)
    bg_img = Image.open(bg_path).convert("RGBA")

    # 将前景图像缩放到背景图像的高度
    bg_height = fg_img.size[1]
    bg_width = bg_height * bg_img.size[0] // bg_img.size[1]
    if bg_width < fg_img.size[0]:
        bg_width = fg_img.size[0]
        bg_height = bg_width * bg_img.size[1] // bg_img.size[0]
    bg_img = bg_img.resize((bg_width, bg_height))

    # 将前景图像叠加到背景图像上
    # 创建一个同背景图相同大小的透明图层
    result_img = Image.new("RGBA", bg_img.size)
    # 将背景图和前景图分别粘贴到透明图层上
    result_img.paste(bg_img, (0, 0))
    # 将前景图像颜色修改为forecolor
    forecolor = tuple(forecolor)
    fg_img = fg_img.convert("RGBA")
    data = fg_img.getdata()
    new_data = []
    for item in data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(forecolor)
    fg_img.putdata(new_data)
    result_img.paste(fg_img, (0, 0), fg_img)  # 使用前景图像作为mask以保留透明度
    l_out = np.array(result_img)

    return l_out

def poisson_blit_images(im_top, im_back, scale_grad=1.0, mode='max'):

    assert np.all(im_top.shape == im_back.shape)

    im_top = im_top.copy().astype(np.float32)
    im_back = im_back.copy().astype(np.float32)
    im_res = np.zeros_like(im_top)

    # frac of gradients which come from source:
    for ch in range(im_top.shape[2]):
        ims = im_top[:, :, ch]
        imd = im_back[:, :, ch]

        [gxs, gys] = get_grads(ims)
        [gxd, gyd] = get_grads(imd)

        gxs *= scale_grad
        gys *= scale_grad

        gxs_idx = gxs != 0
        gys_idx = gys != 0
        # mix the source and target gradients:
        if mode == 'max':
            gx = gxs.copy()
            gxm = (np.abs(gxd)) > np.abs(gxs)
            gx[gxm] = gxd[gxm]

            gy = gys.copy()
            gym = np.abs(gyd) > np.abs(gys)
            gy[gym] = gyd[gym]

            # get gradient mixture statistics:
            f_gx = np.sum((gx[gxs_idx] == gxs[gxs_idx]).flat) / (np.sum(gxs_idx.flat) + 1e-6)
            f_gy = np.sum((gy[gys_idx] == gys[gys_idx]).flat) / (np.sum(gys_idx.flat) + 1e-6)
            if min(f_gx, f_gy) <= 0.35:
                m = 'max'
                if scale_grad > 1:
                    m = 'blend'
                return poisson_blit_images(im_top, im_back, scale_grad=1.5, mode=m)

        elif mode == 'src':
            gx, gy = gxd.copy(), gyd.copy()
            gx[gxs_idx] = gxs[gxs_idx]
            gy[gys_idx] = gys[gys_idx]

        elif mode == 'blend':  # from recursive call:
            # just do an alpha blend
            gx = gxs + gxd
            gy = gys + gyd

        im_res[:, :, ch] = np.clip(poisson_solve(gx, gy, imd), 0, 255)

    return im_res.astype('uint8')


def get_grads(im):

    [H, W] = im.shape
    Dx, Dy = np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)
    j, k = np.atleast_2d(np.arange(0, H - 1)).T, np.arange(0, W - 1)
    Dx[j, k] = im[j, k + 1] - im[j, k]
    Dy[j, k] = im[j + 1, k] - im[j, k]
    return Dx, Dy


def poisson_solve(gx, gy, bnd):

    # convert to double:
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    bnd = bnd.astype(np.float32)

    H, W = bnd.shape
    L = get_laplacian(gx, gy)

    # set the interior of the boundary-image to 0:
    bnd[1:-1, 1:-1] = 0
    # get the boundary laplacian:
    L_bp = np.zeros_like(L)
    L_bp[1:-1, 1:-1] = -4 * bnd[1:-1, 1:-1] \
                       + bnd[1:-1, 2:] + bnd[1:-1, 0:-2] \
                       + bnd[2:, 1:-1] + bnd[0:-2, 1:-1]  # delta-x
    L = L - L_bp
    L = L[1:-1, 1:-1]

    # compute the 2D DST:
    L_dst = DST(DST(L).T).T  # first along columns, then along rows

    # normalize:
    [xx, yy] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    D = (2 * np.cos(np.pi * xx / (W - 1)) - 2) + (2 * np.cos(np.pi * yy / (H - 1)) - 2)
    L_dst = L_dst / D

    img_interior = IDST(IDST(L_dst).T).T  # inverse DST for rows and columns

    img = bnd.copy()

    img[1:-1, 1:-1] = img_interior

    return img


def get_laplacian(Dx, Dy):
    [H, W] = Dx.shape
    Dxx, Dyy = np.zeros((H, W)), np.zeros((H, W))
    j, k = np.atleast_2d(np.arange(0, H - 1)).T, np.arange(0, W - 1)
    Dxx[j, k + 1] = Dx[j, k + 1] - Dx[j, k]
    Dyy[j + 1, k] = Dy[j + 1, k] - Dy[j, k]
    return Dxx + Dyy


def DST(x):
    X = scipy.fftpack.dst(x, type=1, axis=0)
    return X / 2.0


def IDST(X):

    n = X.shape[0]
    x = np.real(scipy.fftpack.idst(X, type=1, axis=0))
    return x / (n + 1.0)
