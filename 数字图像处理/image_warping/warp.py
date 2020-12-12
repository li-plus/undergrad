import cv2
import numpy as np
import math


def get_perspective_transform(src_points, dst_points):
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    assert src_points.shape == (4, 2) and dst_points.shape == (4, 2)

    def _get_std_mat(points):
        points = cv2.convertPointsToHomogeneous(points).squeeze(1)
        A, b = points[:3].T, points[3]
        scale = np.linalg.solve(A, b)
        A *= scale
        return A

    A = _get_std_mat(src_points)
    B = _get_std_mat(dst_points)
    matrix = B.dot(np.linalg.inv(A))
    return matrix


def warp_perspective(src, matrix, dst_shape):
    assert src.ndim == 2 or src.ndim == 3
    src_h, src_w = src.shape[:2]
    dst_w, dst_h = dst_shape[:2]

    inv_warp = np.linalg.inv(matrix)
    dst_indices = np.array(list(np.ndindex((dst_h, dst_w))), dtype=np.int32)
    dst_indices_homo = cv2.convertPointsToHomogeneous(dst_indices).squeeze(1)
    src_indices_homo = inv_warp.dot(dst_indices_homo.T).T
    src_indices = cv2.convertPointsFromHomogeneous(src_indices_homo).squeeze(1)

    src_indices = src_indices.astype(np.int32)
    src_x, src_y = src_indices[:, 0], src_indices[:, 1]

    valid_mask = (0 <= src_x) & (src_x < src_w) & (0 <= src_y) & (src_y < src_h)

    dst_x, dst_y = dst_indices[valid_mask].T
    src_x, src_y = src_indices[valid_mask].T

    dst = np.zeros(dst_shape, dtype=np.uint8)
    dst[dst_y, dst_x] = src[src_y, src_x]
    return dst


def warp_sphere(src, dst_size):
    assert src.ndim == 2 or src.ndim == 3
    src_h, src_w = src.shape[:2]
    dst_w, dst_h = dst_size
    dst_shape = (dst_h, dst_w, src.shape[2]) if src.ndim == 3 else (dst_h, dst_w)
    dst = np.zeros(dst_shape, dtype=np.uint8)

    max_dst_radius = min(dst_w, dst_h) // 2
    dst_center_y, dst_center_x = max_dst_radius, max_dst_radius
    src_center_y, src_center_x = src_h // 2, src_w // 2

    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            dst_radius = math.sqrt((dst_y - dst_center_y) ** 2 + (dst_x - dst_center_x) ** 2)
            if dst_radius > max_dst_radius:
                continue
            src_radius = max_dst_radius * math.asin(dst_radius / max_dst_radius)
            scale_factor = src_radius / dst_radius if dst_radius else 1
            src_y = int(round(scale_factor * (dst_y - dst_center_y) + src_center_y))
            src_x = int(round(scale_factor * (dst_x - dst_center_x) + src_center_x))
            if 0 <= src_y < src_h and 0 <= src_x < src_w:
                dst[dst_y, dst_x] = src[src_y, src_x]
    return dst


def warp_sphere_inv(src, dst_size):
    assert src.ndim == 2 or src.ndim == 3
    src_h, src_w = src.shape[:2]
    dst_w, dst_h = dst_size
    dst_shape = (dst_h, dst_w, src.shape[2]) if src.ndim == 3 else (dst_h, dst_w)
    dst = np.zeros(dst_shape, dtype=np.uint8)

    max_src_radius = min(src_h, src_w) // 2
    dst_center_y, dst_center_x = dst_h // 2, dst_w // 2
    src_center_y, src_center_x = src_h // 2, src_w // 2

    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            dst_radius = math.sqrt((dst_y - dst_center_y) ** 2 + (dst_x - dst_center_x) ** 2)
            if abs(1 - dst_radius > max_src_radius) > 1:
                continue
            src_radius = max_src_radius * math.acos(1 - dst_radius / max_src_radius)
            scale_factor = src_radius / dst_radius if dst_radius else 1
            src_y = int(round(scale_factor * (dst_y - dst_center_y) + src_center_y))
            src_x = int(round(scale_factor * (dst_x - dst_center_x) + src_center_x))
            if 0 <= src_y < src_h and 0 <= src_x < src_w:
                dst[dst_y, dst_x] = src[src_y, src_x]
    return dst
