from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import linalg


def get_neighbors(point):
    i, j = point
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def poisson_solve(src, dst, mask):
    assert src.shape == dst.shape == mask.shape
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool)
    mask_indices = list(zip(*np.nonzero(mask)))
    mask_size = len(mask_indices)

    # construct matrix A
    A = sparse.lil_matrix((mask_size, mask_size), dtype=np.float32)

    pos2idx = {}
    for p_idx, p in enumerate(mask_indices):
        pos2idx[p] = p_idx

    for p_idx, p in enumerate(mask_indices):
        A[p_idx, p_idx] = 4
        for q in get_neighbors(p):
            if mask[q]:
                # q is in the mask
                q_idx = pos2idx[q]
                A[p_idx, q_idx] = -1
    A = A.tocsr()

    # construct vector b
    lap = cv2.Laplacian(src, cv2.CV_32F, borderType=cv2.BORDER_CONSTANT)
    edge_lap = cv2.Laplacian(dst * ~mask, cv2.CV_32F, borderType=cv2.BORDER_CONSTANT)
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


def process(index):
    in_dir = Path('inputs') / str(index)

    mask = cv2.imread(str(in_dir / 'mask.png'), cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(str(in_dir / 'source.jpg'))
    dst = cv2.imread(str(in_dir / 'target.jpg'))

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    out_dir = Path('outputs') / str(index)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_mask = mask.astype(np.bool)[:, :, np.newaxis]
    out = src * bin_mask + dst * ~bin_mask
    cv2.imwrite(str(out_dir / 'naive.jpg'), out)

    out = seamless_clone(src, dst, mask)
    cv2.imwrite(str(out_dir / 'poisson.jpg'), out)


def main():
    for i in range(1, 3):
        process(i)


if __name__ == "__main__":
    main()
