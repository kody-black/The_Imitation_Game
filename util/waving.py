import numpy as np
import math
import random

# 该函数用于实现波浪效果
def waving(srcBmp, char_surf_ls, length, orientation = 'horizontal', level = 2, period=0.5):
    contour_ls = []
    for char_surf in char_surf_ls:
        x, y = np.where(char_surf != 0)
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(x.shape[0], 1)
        cord = np.concatenate([x, y], axis=1)
        contour_ls.append(cord)

    dMultValue = level  # 修改为根据 level 动态确定
    dPhase = random.random() * 2 * math.pi
    height, width = srcBmp.shape
    vertical = False

    if orientation == 'horizontal':
        rand = random.randint(1, length // 2)
        dBaseAxisLen = width / rand
    elif orientation == 'vertical':
        rand = random.randint(1, 2)
        dBaseAxisLen = height / rand
        vertical = True
    elif orientation == 'all':
        if random.random() < 0.5:
            rand = random.randint(1, 2)
            dBaseAxisLen = height / rand
            vertical = True
        else:
            rand = random.randint(1, length // 2)
            dBaseAxisLen = width / rand
    
    destBmp = np.zeros_like(srcBmp)
    new_contour_ls = [[] for _ in range(len(contour_ls))]

    for i in range(width):
        for j in range(height):
            if vertical:
                dx = math.pi * 2 * j / (dBaseAxisLen * period)
            else:
                dx = math.pi * 2 * i / (dBaseAxisLen * period)
            dx += dPhase
            dy = math.sin(dx) * dMultValue  # 使用 level 调整振幅

            if vertical:
                nOldX = int(i + dy)
                nOldY = j
            else:
                nOldX = i
                nOldY = int(j + dy)
            if 0 <= nOldX < width and 0 <= nOldY < height:
                destBmp[nOldY, nOldX] = srcBmp[j, i]

            index = [k for k in range(len(contour_ls)) if np.sum(np.all(contour_ls[k] == [[j, i]], axis=1)) > 0]
            if len(index) > 0:
                for k in index:
                    new_contour_ls[k].append([nOldY, nOldX])

    new_char_surf_ls = []
    for perspected_points in new_contour_ls:
        perspected_points = np.array(perspected_points)
        new_char_surf = np.zeros((char_surf_ls[0].shape[0], char_surf_ls[0].shape[1])).astype(np.uint8)
        perspected_points = perspected_points.astype(np.int64)
        new_char_surf[perspected_points[:, 0], perspected_points[:, 1]] = 255
        new_char_surf_ls.append(new_char_surf)
    
    return destBmp, new_char_surf_ls
