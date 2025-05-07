import numpy as np
import math
import random


def waving(srcBmp, char_surf_ls, length):
    contour_ls = []
    for char_surf in char_surf_ls:
        x, y = np.where(char_surf != 0)
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(x.shape[0], 1)
        cord = np.concatenate([x, y], axis=1)
        contour_ls.append(cord)

    dMultValue = random.randint(4, 8)
    dPhase = random.random() * 2 * math.pi
    height, width = srcBmp.shape

    if random.random() < 0.5:
        vertical = True
        rand = random.randint(1, 2)
        dBaseAxisLen = height / rand
    else:
        vertical = False
        rand = random.randint(1, length // 2)
        dBaseAxisLen = width / rand
    destBmp = np.zeros_like(srcBmp)

    new_contour_ls = [[] for _ in range(len(contour_ls))]
    for i in range(width):
        for j in range(height):
            if vertical:
                dx = math.pi * 2 * j / dBaseAxisLen
            else:
                dx = math.pi * 2 * i / dBaseAxisLen
            dx += dPhase
            dy = math.sin(dx)

            if vertical:
                nOldX = int(i + dy * dMultValue)
                nOldY = j
            else:
                nOldX = i
                nOldY = int(j + dy * dMultValue)
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
