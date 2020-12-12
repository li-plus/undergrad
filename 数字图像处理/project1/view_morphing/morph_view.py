import numpy as np
from scipy.spatial import Delaunay
import cv2
from face_morphing import morph


def warp_points(src_points, mat):
    src_points_homo = cv2.convertPointsToHomogeneous(src_points).squeeze(1)
    dst_points_homo = mat.dot(src_points_homo.T).T
    dst_points = cv2.convertPointsFromHomogeneous(dst_points_homo).squeeze(1)
    return dst_points


def warp_delaunay_view(src, src_points_mid, dst_points_mid, prewarp, postwarp):
    """
        prewarp              warp              postwarp
    src  ---->    src_mid    --->    dst_mid    ----->  dst
              src_points_mid ---> dst_points_mid
    """
    delaunay = Delaunay(dst_points_mid)

    inv_prewarp = np.linalg.inv(prewarp)
    inv_postwarp = np.linalg.inv(postwarp)

    inv_mats = morph.get_triangular_affine_inverse(src_points_mid, dst_points_mid, delaunay.simplices)
    inv_mats = [inv_prewarp.dot(m).dot(inv_postwarp) for m in inv_mats]

    h, w = src.shape[:2]
    dst_coords = np.array(list(np.ndindex(w, h)), dtype=np.int32)
    dst_coords_mid = warp_points(dst_coords, inv_postwarp)
    dst_tri_indices = delaunay.find_simplex(dst_coords_mid)

    dst = morph.warp_triangular(src, src.shape, dst_coords, dst_tri_indices, inv_mats)
    return dst


def morph_view(src, src_points, dst, dst_points, ratio, src_prewarp, dst_prewarp):
    h, w = src.shape[:2]
    corners = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)], dtype=np.float32)
    src_points = np.vstack((src_points, corners)).astype(np.float32)
    dst_points = np.vstack((dst_points, corners)).astype(np.float32)

    src_points = warp_points(src_points, src_prewarp)
    dst_points = warp_points(dst_points, dst_prewarp)

    mid_points = (src_points * (1 - ratio) + dst_points * ratio)
    mid_points, mid_corners = mid_points[:-4], mid_points[-4:]

    postwarp = cv2.getPerspectiveTransform(mid_corners, corners)

    src_mid = warp_delaunay_view(src, src_points, mid_points, src_prewarp, postwarp)
    dst_mid = warp_delaunay_view(dst, dst_points, mid_points, dst_prewarp, postwarp)

    out = (src_mid * (1 - ratio) + dst_mid * ratio).astype(np.uint8)
    return out
