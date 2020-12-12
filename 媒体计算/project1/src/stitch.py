import argparse
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

import poisson
from graphcut import Rect, Point, get_dist_map, get_dist, INF


def get_patch_mask(src: np.ndarray,
                   src_mask: np.ndarray,
                   dst: np.ndarray,
                   dst_mask: np.ndarray
                   ) -> np.ndarray:
    assert src.ndim == dst.ndim == 3
    assert src.shape[:2] == dst.shape[:2] == src_mask.shape == dst_mask.shape
    src_mask = src_mask.astype(np.bool)
    dst_mask = dst_mask.astype(np.bool)

    dst_h, dst_w = dst_mask.shape
    bound_rect = Rect(0, 0, dst_w, dst_h)

    overlap_mask = dst_mask & src_mask
    if not overlap_mask.any() or (overlap_mask == dst_mask).all():
        # without overlap or patch completely covers source image
        return src_mask  # directly take the patch

    if (overlap_mask == src_mask).all():
        # already filled
        return np.zeros_like(src_mask)

    overlap_rows, overlap_cols = overlap_mask.nonzero()
    # we consider top & left neighbors of each pixel, so upper bounds should +2
    overlap_rect = Rect(overlap_cols.min(), overlap_rows.min(),
                        overlap_cols.max() + 2, overlap_rows.max() + 2)
    overlap_rect = overlap_rect.clip(bound_rect)

    G = nx.Graph()
    super_src = Point(-100, -100)
    super_dst = Point(-200, -200)

    patch = src.astype(np.float32)

    dist = get_dist_map(dst, patch)
    assert dist.shape == dst_mask.shape

    for curr in overlap_rect.points:
        nbrs = [curr.top_nbr, curr.left_nbr]
        for nbr in nbrs:
            if nbr not in bound_rect:
                # if neighbor is out of range, ignore
                continue
            if overlap_mask[curr.idx] and overlap_mask[nbr.idx]:
                # both pixels are in overlap area
                cap = get_dist(dist, curr, nbr)
                G.add_edge(curr, nbr, capacity=cap)
            elif overlap_mask[curr.idx] ^ overlap_mask[nbr.idx]:
                # boundary detected, add source / sink edge
                if overlap_mask[curr.idx]:
                    # current pixel inside, neighbor one outside
                    inner_pos, outer_pos = curr, nbr
                else:
                    # neighbor pixel inside, current one outside
                    inner_pos, outer_pos = nbr, curr

                if dst_mask[outer_pos.idx]:
                    # outer pixel only in dst: output-side boundary
                    G.add_edge(inner_pos, super_dst, capacity=INF)
                elif src_mask[outer_pos.idx]:
                    # outer pixel only in patch: patch-side boundary
                    G.add_edge(inner_pos, super_src, capacity=INF)
                else:
                    # outer pixel on neither dst nor patch: not a boundary
                    pass

    patch_mask = src_mask.copy()
    if G.has_node(super_src) and G.has_node(super_dst):
        max_flow, partition = nx.minimum_cut(G, super_src, super_dst)
        patch_side, dst_side = partition
        for pos in dst_side:
            patch_mask[pos.idx] = False

    return patch_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../output/golden_align')
    parser.add_argument('--out', type=str, default='out.jpg')
    args = parser.parse_args()

    in_dir = Path(args.dir)
    img_paths = sorted(in_dir.glob('*.jpg'))
    assert len(img_paths) > 1

    imgs = []
    masks = []
    for img_path in img_paths:
        mask_path = in_dir / f'{img_path.stem}.png'
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None and mask is not None
        imgs.append(img)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask.astype(np.bool))

    dst = imgs[0]
    dst_mask = masks[0]

    for idx, (src, src_mask) in enumerate(zip(imgs[1:], masks[1:])):
        assert dst_mask.shape == src_mask.shape \
               == src.shape[:2] == dst.shape[:2]

        patch_mask = get_patch_mask(src, src_mask, dst, dst_mask)

        out = src.copy()
        out[dst_mask] = dst[dst_mask]
        out = poisson.seamless_clone(src, out, patch_mask)
        dst_mask |= src_mask
        dst = out * dst_mask[..., np.newaxis]
        assert cv2.imwrite(f'{idx}.jpg', dst)
        print('Done', idx)

    assert cv2.imwrite(args.out, dst)


if __name__ == '__main__':
    main()
