import numpy as np
import cv2
from scipy.ndimage import filters


def get_energy_map(src):
    assert src.ndim == 3

    filter_dy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=np.int32)
    filter_dy = np.stack([filter_dy] * 3, axis=2)

    filter_dx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.int32)
    filter_dx = np.stack([filter_dx] * 3, axis=2)

    src = src.astype(np.int32)
    energy_map = np.abs(filters.convolve(src, filter_dy)) + np.abs(filters.convolve(src, filter_dx))

    energy_map = np.sum(energy_map, axis=2)
    return energy_map


def get_cost_map(energy_map):
    h, w = energy_map.shape
    cost_map = energy_map.copy().astype(np.int32)
    parent_map = np.zeros((h, w), dtype=np.int32)

    for r in range(1, h):
        for c in range(w):
            if c == 0:
                min_c = np.argmin(cost_map[r - 1, :2])
            else:
                min_c = np.argmin(cost_map[r - 1, c - 1: c + 2]) + c - 1

            parent_map[r, c] = min_c
            cost_map[r, c] += cost_map[r - 1, min_c]

    return cost_map, parent_map


def get_seam(cost_map, parent_map):
    h, w = cost_map.shape
    c = np.argmin(cost_map[-1])
    indices = [c]
    mask = np.ones((h, w), dtype=np.bool)

    for r in range(h - 1, -1, -1):
        indices.append(c)
        mask[r, c] = False
        c = parent_map[r, c]

    indices.reverse()
    return indices, mask


def remove_seam(src, seam_mask):
    assert src.ndim in [2, 3]

    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.stack([seam_mask] * c, axis=2)
        dst = src[seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[seam_mask].reshape((h, w - 1))

    return dst


def insert_seam(src, seam):
    h, w, c = src.shape
    dst = np.empty((h, w + 1, c), dtype=np.uint8)

    for r in range(h):
        c = seam[r] if seam[r] else 1

        dst[r, :c] = src[r, :c]
        dst[r, c] = np.mean(src[r, c-1:c+1], axis=0).astype(np.uint8)
        dst[r, c + 1:] = src[r, c:]

    return dst


def remove_seams(src, num_seams):
    for _ in range(num_seams):
        energy_map = get_energy_map(src)
        cost_map, parent_map = get_cost_map(energy_map)
        _, seam_mask = get_seam(cost_map, parent_map)
        src = remove_seam(src, seam_mask)

    return src


def insert_seams(src, num_seams):
    seams = []
    tmp_src = src.copy()

    for _ in range(num_seams):
        energy_map = get_energy_map(tmp_src)
        cost_map, parent_map = get_cost_map(energy_map)
        indices, mask = get_seam(cost_map, parent_map)
        tmp_src = remove_seam(tmp_src, mask)
        seams.append(indices)

    seams.reverse()

    for _ in range(num_seams):
        seam = seams.pop()
        src = insert_seam(src, seam)
        for remain in seams:
            remain[remain >= seam] += 2

    return src


def remove_object(src, mask):
    mask_energy = -0xffff
    mask = mask > 0

    while mask.any():
        energy_map = get_energy_map(src)
        energy_map[mask] = mask_energy
        cost_map, parent_map = get_cost_map(energy_map)
        _, seam_mask = get_seam(cost_map, parent_map)
        src = remove_seam(src, seam_mask)
        mask = remove_seam(mask, seam_mask)

    return src
