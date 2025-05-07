# coding=utf-8
import random
import numpy as np
import math
import cv2

def perspective(img, contours_ls, cfg):
    # 根据配置生成透视变换矩阵H。
    H = get_H_matrix(cfg)

    # 计算原始图像的中心点。
    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)
    # 定义原始图像四个角的坐标。
    points = np.ones((3, 4), dtype=np.float32)
    points[:2, 0] = np.array([0, 0], dtype=np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype=np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype=np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype=np.float32).T

    # 使用透视变换矩阵变换四角坐标，并基于变换后的坐标计算新画布的大小。
    perspected_points = center_pointsPerspective(points, H, img_center)
    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]
    
    # 根据变换后的点坐标，计算新画布的宽度和高度。
    canvas_w = int(2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]),
                           np.max(perspected_points[0, :]) - img_center[0])) + 10
    canvas_h = int(2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]),
                           np.max(perspected_points[1, :]) - img_center[1])) + 10
    
    # 对整个画布应用透视变换
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tly = (canvas_h - img_h) // 2
    tlx = (canvas_w - img_w) // 2
    canvas[tly:tly + img_h, tlx:tlx + img_w] = img

    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    canvas = center_warpPerspective(canvas, H, canvas_center, canvas_size)

    # 对每个文字的轮廓坐标应用相同的透视变换，计算变换后的新位置。
    bbs = []  # 存储变换后的文字边界框
    perspected_contour_ls = []  # 存储变换后的文字轮廓坐标
    for contour in contours_ls:
        original_points = np.ones((3, contour.shape[0]), dtype=np.float32)
        contour = np.squeeze(contour, axis=1).T
        contour = np.roll(contour, -1, axis=0)
        original_points[:2, :] = contour
        original_points[0, :] += tlx
        original_points[1, :] += tly
        new_perspected_points = center_pointsPerspective(original_points, H, canvas_center)
        new_perspected_points[0, :] /= new_perspected_points[2, :]
        new_perspected_points[1, :] /= new_perspected_points[2, :]
        new_perspected_points = new_perspected_points[:2, :].T

        left, right = np.min(new_perspected_points[:, 0]), np.max(new_perspected_points[:, 0])
        top, bottom = np.min(new_perspected_points[:, 1]), np.max(new_perspected_points[:, 1])
        bbs.append([math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)])
        perspected_contour_ls.append(new_perspected_points)

    bbs = np.array(bbs)
    left = np.min(bbs[:, 0])
    top = np.min(bbs[:, 1])
    right = np.max(bbs[:, 2]) + 1
    bottom = np.max(bbs[:, 3]) + 1
    width = right - left
    height = bottom - top
    padding = [random.randint(cfg["padding_min"], cfg["padding_max"]) for _ in range(4)]

    # 根据变换后的文字位置，确定最终图像的尺寸和添加padding
    resimg = np.zeros((height + padding[0] + padding[1], width + padding[2] + padding[3])).astype(np.uint8)
    # Before:
    # resimg[padding[0]:padding[0] + height, padding[2]:padding[2] + width] = canvas[top:bottom, left:right]

    # After:
    height, width = canvas[top:bottom, left:right].shape
    resimg[padding[0]:padding[0] + height, padding[2]:padding[2] + width] = canvas[top:bottom, left:right]

    for perspected_points in perspected_contour_ls:
        perspected_points[:, 0] += padding[2] - left
        perspected_points[:, 1] += padding[0] - top

    char_surf_ls = []
    for perspected_points in perspected_contour_ls:
        char_surf = np.zeros((resimg.shape[0], resimg.shape[1])).astype(np.uint8)
        perspected_points = perspected_points.astype(np.int64)
        char_surf[perspected_points[:, 1], perspected_points[:, 0]] = 255
        char_surf_ls.append(char_surf)

    return resimg, char_surf_ls


def get_H_matrix(cfg):
    rotate_angle = cfg["rotate_param"][0] * np.random.randn() + cfg["rotate_param"][1]
    zoom = cfg["zoom_param"][0] * np.random.randn(2) + cfg["zoom_param"][1]
    shear_angle = cfg["shear_param"][0] * np.random.randn(2) + cfg["shear_param"][1]
    perspect = cfg["perspect_param"][0] * np.random.randn(2) + cfg["perspect_param"][1]

    rotate_angle = rotate_angle * math.pi / 180.
    shear_x_angle = shear_angle[0] * math.pi / 180.
    shear_y_angle = shear_angle[1] * math.pi / 180.
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect

    H_scale = np.array([[scale_w, 0, 0],
                        [0, scale_h, 0],
                        [0, 0, 1]], dtype=np.float32)
    H_rotate = np.array([[math.cos(rotate_angle), math.sin(rotate_angle), 0],
                         [-math.sin(rotate_angle), math.cos(rotate_angle), 0],
                         [0, 0, 1]], dtype=np.float32)
    H_shear = np.array([[1, math.tan(shear_x_angle), 0],
                        [math.tan(shear_y_angle), 1, 0],
                        [0, 0, 1]], dtype=np.float32)
    H_perspect = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [perspect_x, perspect_y, 1]], dtype=np.float32)

    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    return H


def center_pointsPerspective(points, H, center):
    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    return M.dot(points)


def center_warpPerspective(img, H, center, size):
    P = np.array([[1, 0, center[0]],
                  [0, 1, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return img
