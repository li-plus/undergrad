import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import linalg


def poisson_solve(src, dst, mask):
    assert src.shape == dst.shape == mask.shape
    h, w = src.shape
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool)
    mask_indices = list(zip(*np.nonzero(mask)))
    mask_size = len(mask_indices)

    # construct matrix A
    A = sparse.lil_matrix((mask_size, mask_size), dtype=np.float32)

    pos2idx = {}
    for p_idx, (pi, pj) in enumerate(mask_indices):
        pos2idx[pi, pj] = p_idx

    for p_idx, (pi, pj) in enumerate(mask_indices):
        A[p_idx, p_idx] = 4
        nbrs = [(pi, pj - 1), (pi, pj + 1), (pi - 1, pj), (pi + 1, pj)]
        for qi, qj in nbrs:
            if 0 <= qi < h and 0 <= qj < w and mask[qi, qj]:
                q_idx = pos2idx[qi, qj]
                A[p_idx, q_idx] = -1

    A = A.tocsr()

    # construct vector b
    lap = cv2.Laplacian(src, cv2.CV_32F, borderType=cv2.BORDER_CONSTANT)
    edge_lap = cv2.Laplacian(dst * ~mask, cv2.CV_32F,
                             borderType=cv2.BORDER_CONSTANT)
    b = (edge_lap - lap)[mask]

    # solve Ax = b
    x = linalg.spsolve(A, b)

    # merge
    out = dst.copy()
    out[mask] = x

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def seamless_clone(src, dst, mask):
    assert src.ndim == dst.ndim == 3
    assert src.shape[:2] == dst.shape[:2] == mask.shape
    _, _, channels = src.shape
    out = np.zeros(src.shape, dtype=np.uint8)
    for c in range(channels):
        out[:, :, c] = poisson_solve(src[:, :, c], dst[:, :, c], mask)
    return out
