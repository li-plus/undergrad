import numpy as np
from scipy.spatial import Delaunay
import cv2


def interpolate_sparse(src, coords):
    x0, y0 = coords.astype(np.int32)
    dx, dy = coords - coords.astype(np.int32)
    dx = dx[:, None]
    dy = dy[:, None]

    left = src[y0, x0] * (1 - dy) + src[y0 + 1, x0] * dy
    right = src[y0, x0 + 1] * (1 - dy) + src[y0 + 1, x0 + 1] * dy
    mid_values = left * (1 - dx) + right * dx

    return mid_values


def get_triangular_affine_inverse(src_points, dst_points, dst_simplices):
    affine_inv_mats = []

    for tri in dst_simplices:
        src_tri = np.vstack([src_points[tri].T, [1, 1, 1]])
        dst_tri = np.vstack([dst_points[tri].T, [1, 1, 1]])
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))
        affine_inv_mats.append(mat)

    return affine_inv_mats


def warp_triangular(src, dst_shape, dst_coords, tri_indices, inv_mats):
    h, w = src.shape[:2]
    dst = np.zeros(dst_shape, dtype=np.uint8)

    for tri_index, inv_mat in enumerate(inv_mats):
        if (tri_indices == tri_index).any():
            tri_dst_coords = dst_coords[tri_indices == tri_index]
            tri_dst_coords_homo = cv2.convertPointsToHomogeneous(tri_dst_coords).squeeze(1)

            tri_src_coords_homo = np.dot(inv_mat, tri_dst_coords_homo.T).T
            tri_src_coords = cv2.convertPointsFromHomogeneous(tri_src_coords_homo).squeeze(1)

            src_x, src_y = tri_src_coords.T
            valid_indices = (0 <= src_x) & (src_x < w-1) & (0 <= src_y) & (src_y < h-1)
            dst_x, dst_y = tri_dst_coords[valid_indices].T
            dst[dst_y, dst_x] = interpolate_sparse(src, tri_src_coords[valid_indices].T)

    return dst


def warp_delaunay(src, src_points, dst_points):
    delaunay = Delaunay(dst_points)

    inv_mats = get_triangular_affine_inverse(src_points, dst_points, delaunay.simplices)

    h, w = src.shape[:2]
    dst_coords = np.array(list(np.ndindex(w, h)), dtype=np.int32)
    dst_tri_indices = delaunay.find_simplex(dst_coords)

    dst = warp_triangular(src, src.shape, dst_coords, dst_tri_indices, inv_mats)
    return dst


def morph_image(src, src_points, dst, dst_points, ratio):
    mid_points = (src_points * (1 - ratio) + dst_points * ratio).astype(np.int32)

    src_mid = warp_delaunay(src, src_points, mid_points)
    dst_mid = warp_delaunay(dst, dst_points, mid_points)

    out = (src_mid * (1 - ratio) + dst_mid * ratio).astype(np.uint8)
    return out
