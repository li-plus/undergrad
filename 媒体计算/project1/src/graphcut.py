from __future__ import annotations

import random
import warnings
from typing import NamedTuple, Tuple, Optional

import networkx as nx
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage.color import rgb2gray

INF = 1e8
EPS = 1e-8

LEFT = 0
TOP = 1

SizeType = Tuple[int, int]
PointLike = Tuple[int, int]


class Point(NamedTuple):
    x: int
    y: int

    @property
    def idx(self):
        return self.y, self.x

    def __add__(self, other: PointLike) -> Point:
        dx, dy = other
        return Point(self.x + dx, self.y + dy)

    def __sub__(self, other: PointLike) -> Point:
        dx, dy = other
        return Point(self.x - dx, self.y - dy)

    @property
    def left_nbr(self):
        return Point(self.x - 1, self.y)

    @property
    def right_nbr(self):
        return Point(self.x + 1, self.y)

    @property
    def top_nbr(self):
        return Point(self.x, self.y - 1)

    @property
    def bottom_nbr(self):
        return Point(self.x, self.y + 1)


class Rect(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def slice(self) -> Tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    @property
    def points(self):
        for y in range(self.y1, self.y2):
            for x in range(self.x1, self.x2):
                yield Point(x, y)

    def clip(self, rect: Rect) -> Rect:
        x1 = max(self.x1, rect.x1)
        y1 = max(self.y1, rect.y1)
        x2 = min(self.x2, rect.x2)
        y2 = min(self.y2, rect.y2)
        return Rect(x1, y1, x2, y2)

    def __add__(self, other: PointLike) -> Rect:
        dx, dy = other
        x1, y1, x2, y2 = self
        return Rect(x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def __sub__(self, other: PointLike):
        dx, dy = other
        return self + (-dx, -dy)

    def __contains__(self, item: PointLike) -> bool:
        x, y = item
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

    @staticmethod
    def from_center_size(center: PointLike, size: SizeType) -> Rect:
        w, h = size
        cx, cy = center
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h
        return Rect(x1, y1, x2, y2)


class PixelGrid(object):
    def __init__(self, size: SizeType):
        w, h = size
        self.has_left = np.zeros((h, w), dtype=np.bool)
        self.has_top = np.zeros((h, w), dtype=np.bool)
        self.cost_left = np.zeros((h, w), dtype=np.float32)
        self.cost_top = np.zeros((h, w), dtype=np.float32)
        self.src_off = np.zeros((h, w, 2), dtype=np.int32)


def get_grad(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = src.astype(np.float32)
    src_dy, src_dx = np.gradient(rgb2gray(src), axis=(0, 1))
    src_dy = np.abs(src_dy) / 255.
    src_dx = np.abs(src_dx) / 255.
    return src_dx, src_dy


def get_dist_map(dst: np.ndarray, patch: np.ndarray) -> np.ndarray:
    dst = dst.astype(np.float32)
    patch = patch.astype(np.float32)
    dist = np.square(dst - patch).sum(axis=-1)
    return dist


def get_pair_dist(img1: np.ndarray, img2: np.ndarray,
                  pos1: Point, pos2: Point) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    dist = (np.square(img1[pos1.idx] - img2[pos1.idx]) +
            np.square(img1[pos2.idx] - img2[pos2.idx])).sum(-1)
    # TODO: use grad?
    return float(dist)


def get_dist(dist: np.ndarray,
             dst_pos: Point,
             patch_pos: Point,
             dst_grad: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             patch_grad: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             ) -> float:
    cost = dist[dst_pos.idx] + dist[patch_pos.idx]

    if dst_grad is not None and patch_grad is not None:
        dst_dx, dst_dy = dst_grad
        patch_dx, patch_dy = patch_grad
        if dst_pos.x == patch_pos.x:
            # vertical gradient
            d_dst = dst_dy
            d_patch = patch_dy
        else:
            # horizontal gradient
            d_dst = dst_dx
            d_patch = patch_dx

        cost /= (d_dst[dst_pos.idx] + d_dst[patch_pos.idx] +
                 d_patch[dst_pos.idx] + d_patch[patch_pos.idx] + 1)

    return float(cost)


def get_seam_node_pos(pos1: Point, pos2: Point, dst: np.ndarray) -> Point:
    dst_h, dst_w, _ = dst.shape
    ravel_pos1 = np.ravel_multi_index((pos1.y, pos1.x), (dst_h, dst_w))
    ravel_pos2 = np.ravel_multi_index((pos2.y, pos2.x), (dst_h, dst_w))
    return Point(ravel_pos1 + dst.size, ravel_pos2 + dst.size)


def blend(dst: np.ndarray,
          dst_mask: np.ndarray,
          patch: np.ndarray,
          patch_mask: np.ndarray,
          src: np.ndarray,
          offset: Point,
          grid: Optional[PixelGrid]
          ) -> np.ndarray:
    assert dst.shape == patch.shape
    assert dst.shape[:2] == dst_mask.shape == patch_mask.shape
    update_seam = grid is not None
    dst_h, dst_w = dst_mask.shape
    bound_rect = Rect(0, 0, dst_w, dst_h)

    overlap_mask = dst_mask & patch_mask
    if not overlap_mask.any() or (overlap_mask == dst_mask).all():
        # without overlap or patch completely covers source image
        dst[patch_mask] = patch[patch_mask]  # directly take the patch
        # update grid nodes
        if update_seam:
            grid.src_off[bound_rect.slice] = offset
        return dst

    if (overlap_mask == patch_mask).all():
        # already filled
        print('Attempt to fill a fully-patched area -> Refinement')

    overlap_rows, overlap_cols = overlap_mask.nonzero()
    # we consider top & left neighbors of each pixel, so upper bounds should +2
    overlap_rect = Rect(overlap_cols.min(), overlap_rows.min(),
                        overlap_cols.max() + 2, overlap_rows.max() + 2)
    overlap_rect = overlap_rect.clip(bound_rect)

    G = nx.Graph()
    super_src = Point(-100, -100)
    super_dst = Point(-200, -200)

    patch = patch.astype(np.float32)

    # dst_grad = get_grad(dst)
    # patch_grad = get_grad(patch)
    dist = get_dist_map(dst, patch)
    assert dist.shape == dst_mask.shape

    for curr in overlap_rect.points:
        nbrs = [curr.top_nbr, curr.left_nbr]
        directions = [TOP, LEFT]
        for nbr, direction in zip(nbrs, directions):
            if nbr not in bound_rect:
                # if neighbor is out of range, ignore
                continue
            if overlap_mask[curr.idx] and overlap_mask[nbr.idx]:
                # both pixels are in overlap area
                if update_seam and (
                        (direction == LEFT and grid.has_left[curr.idx]) or
                        (direction == TOP and grid.has_top[curr.idx])):
                    # old cut detected, need to insert a seam node
                    #                       src
                    #                        |
                    #                     cap_src
                    #                        |
                    # prev_px--cap_prev--seam_node--cap_curr--curr_px
                    nbr_src_offset = Point(*grid.src_off[nbr.idx])
                    curr_src_offset = Point(*grid.src_off[curr.idx])
                    bg = np.zeros_like(dst)
                    nbr_patch = paste(src, bg, nbr_src_offset)
                    assert (dst[nbr.idx] == nbr_patch[nbr.idx]).all()
                    curr_patch = paste(src, bg, curr_src_offset)
                    assert (dst[curr.idx] == curr_patch[curr.idx]).all()
                    cap_src = get_pair_dist(nbr_patch, curr_patch, nbr, curr)
                    cap_prev = get_pair_dist(nbr_patch, patch, nbr, curr)
                    cap_curr = get_pair_dist(curr_patch, patch, nbr, curr)
                    seam_node = get_seam_node_pos(nbr, curr, dst)
                    G.add_edge(nbr, seam_node, capacity=cap_prev)
                    G.add_edge(curr, seam_node, capacity=cap_curr)
                    G.add_edge(super_src, seam_node, capacity=cap_src)
                else:
                    # no old cut occurs, add a normal edge
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
                elif patch_mask[outer_pos.idx]:
                    # outer pixel only in patch: patch-side boundary
                    G.add_edge(inner_pos, super_src, capacity=INF)
                else:
                    # outer pixel on neither dst nor patch: not a boundary
                    pass

    if G.has_node(super_src) and G.has_node(super_dst):
        max_flow, partition = nx.minimum_cut(G, super_src, super_dst)
        patch_side, dst_side = partition
        for pos in patch_side:
            assert isinstance(pos, Point)
            if pos not in bound_rect:
                # got seam node or source/sink, ignore
                continue
            # copy patch-side pixels to overlapping area
            dst[pos.idx] = patch[pos.idx]

            if update_seam:
                # update the pixel pos in src coord system
                grid.src_off[pos.idx] = offset

                # note down this seam, update has_left and cost_left
                for top, curr in [(pos.top_nbr, pos), (pos, pos.bottom_nbr)]:
                    if not (top in bound_rect and curr in bound_rect):
                        continue
                    if (top in dst_side and curr in patch_side) or (
                            top in patch_side and curr in dst_side):
                        # top nbr in opposite side
                        grid.has_top[curr.idx] = True
                        if G.has_edge(top, curr):
                            # if they are linked, update its cost,
                            cap = G.get_edge_data(curr, top)['capacity']
                        else:
                            # there is a seam node between them
                            seam_pos = get_seam_node_pos(top, curr, dst)
                            cut_pos = curr if seam_pos in dst_side else top
                            cap = G.get_edge_data(seam_pos, cut_pos)['capacity']
                        grid.cost_top[curr.idx] = cap
                    else:
                        grid.has_top[curr.idx] = False
                        grid.cost_top[curr.idx] = 0

                for left, curr in [(pos.left_nbr, pos), (pos, pos.right_nbr)]:
                    if not (left in bound_rect and curr in bound_rect):
                        continue
                    if (left in dst_side and curr in patch_side) or (
                            left in patch_side and curr in dst_side):
                        grid.has_left[curr.idx] = True
                        if G.has_edge(left, curr):
                            cap = G.get_edge_data(curr, left)['capacity']
                        else:
                            seam_pos = get_seam_node_pos(left, curr, dst)
                            cut_pos = curr if seam_pos in dst_side else left
                            cap = G.get_edge_data(seam_pos, cut_pos)['capacity']
                        grid.cost_left[curr.idx] = cap
                    else:
                        grid.has_left[curr.idx] = False
                        grid.cost_left[curr.idx] = 0

        only_patch = patch_mask & ~dst_mask  # exclude dst from patch
        dst[only_patch] = patch[only_patch]  # copy remaining patch pixels
        if update_seam:
            for y, x in zip(*only_patch.nonzero()):
                pos = Point(x, y)
                grid.src_off[pos.idx] = offset

    return dst.astype(np.uint8)


def first_zero_pos(mask: np.ndarray) -> Point:
    rows, cols = np.where(mask == 0)
    assert len(rows) > 0 and len(cols) > 0
    x, y = cols[0], rows[0]
    return Point(x, y)


def get_random_offset(dst_mask: np.ndarray, patch_size: SizeType) -> Point:
    first_pos = first_zero_pos(dst_mask)
    patch_w, patch_h = patch_size
    ctr = first_pos - (random.randint(-patch_w // 4, patch_w // 4),
                       random.randint(-patch_h // 4, patch_h // 4))
    offset = ctr - (patch_w // 2, patch_h // 2)
    return offset


def get_ssd_cost(dst: np.ndarray,
                 dst_mask: np.ndarray,
                 src: np.ndarray) -> np.ndarray:
    warnings.warn('This method is deprecated in favor of get_ssd_cost_fft')
    dst = dst.astype(np.float32) / 255.
    src = src.astype(np.float32) / 255.

    dst_h, dst_w, _ = dst.shape
    patch_h, patch_w, _ = src.shape

    bound_rect = Rect(0, 0, dst_w, dst_h)
    ctr_rect = Rect.from_center_size(first_zero_pos(dst_mask),
                                     (patch_w // 2, patch_h // 2))
    search_rect = ctr_rect - (patch_w // 2, patch_h // 2)

    trans_cost = np.full((dst_h + patch_h - 1, dst_w + patch_w - 1), INF,
                         dtype=np.float32)
    for off_pos in search_rect.points:
        off_rect = Rect(off_pos.x, off_pos.y,
                        off_pos.x + patch_w, off_pos.y + patch_h)
        dst_rect = off_rect.clip(bound_rect)
        dst_mask_block = dst_mask[dst_rect.slice]
        dst_block = dst[dst_rect.slice]

        patch_rect = dst_rect - off_pos
        patch_block = src[patch_rect.slice]

        assert dst_block.shape == patch_block.shape

        if not dst_mask_block.any():
            cost = INF
        else:
            cost = np.square(dst_block[dst_mask_block] -
                             patch_block[dst_mask_block]).mean()
        trans_cost[off_rect.y2 - 1, off_rect.x2 - 1] = cost

    return trans_cost


def get_ssd_cost_fft(dst: np.ndarray,
                     dst_mask: np.ndarray,
                     src: np.ndarray,
                     mode: str) -> np.ndarray:
    # C(t) = \sum_p{I^2(p-t)} + \sum_p{O^2(p)} - 2\sum_p{I(p-t) O(p)}
    assert dst.shape != src.shape
    assert dst.shape[:2] == dst_mask.shape
    dst_mask = dst_mask.astype(np.float64)
    src_h, src_w, channels = src.shape
    dst = dst.astype(np.float64) / 255.
    src = src.astype(np.float64) / 255.

    src_mask = np.ones((src_h, src_w), dtype=np.float64)

    dst_square = np.square(dst).sum(-1)
    dst_square = fftconvolve(dst_square, src_mask[::-1, ::-1], mode)

    src_square = np.square(src).sum(-1)
    src_square = fftconvolve(dst_mask, src_square[::-1, ::-1], mode)

    cross = sum(fftconvolve(dst[:, :, c], src[::-1, ::-1, c], mode)
                for c in range(channels))

    area = fftconvolve(dst_mask, src_mask[::-1, ::-1], mode)
    area[area < EPS] = EPS
    dist_map = src_square + dst_square - 2 * cross
    dist_map[dist_map < EPS] = 0
    dist_map /= area
    return dist_map


def get_entire_patch_offset(
        dst: np.ndarray,
        dst_mask: np.ndarray,
        src: np.ndarray
) -> Point:
    assert dst.shape[:2] == dst_mask.shape
    if not dst_mask.any():
        return Point(0, 0)

    patch_h, patch_w, _ = src.shape

    ctr_rect = Rect.from_center_size(first_zero_pos(dst_mask),
                                     (patch_w // 2, patch_h // 2))
    search_rect = ctr_rect - (patch_w // 2, patch_h // 2)

    trans_cost = get_ssd_cost_fft(dst, dst_mask, src, mode='full')

    # Set the cost outside of RoI (Region of Interest) as INF
    inf_mask = np.ones(trans_cost.shape, dtype=np.bool)
    inf_mask[(search_rect + (patch_w - 1, patch_h - 1)).slice] = False
    trans_cost[inf_mask] = INF

    # Different from paper, here we directly select the minimum cost
    # instead of a weighted random choice
    idx = np.argmin(trans_cost)

    off_y2, off_x2 = np.unravel_index(idx, trans_cost.shape)
    off_bottom_right = Point(int(off_x2), int(off_y2))
    offset = off_bottom_right - (patch_w - 1, patch_h - 1)
    return offset


def get_offset(dst: np.ndarray, dst_mask: np.ndarray,
               src: np.ndarray, placement: str) -> Point:
    if placement == 'random':
        src_h, src_w, _ = src.shape
        offset = get_random_offset(dst_mask, (src_w, src_h))
    elif placement == 'entire-patch':
        offset = get_entire_patch_offset(dst, dst_mask, src)
    else:
        raise ValueError(f'Invalid placement {placement}')
    return offset


def paste(src: np.ndarray, dst: np.ndarray, offset: PointLike) -> np.ndarray:
    src_im = Image.fromarray(src)
    dst_im = Image.fromarray(dst)
    dst_im.paste(src_im, tuple(offset))
    return np.asarray(dst_im)


def get_error_region(region_size: SizeType,
                     grid: PixelGrid) -> Rect:
    err_w, err_h = region_size
    region_mask = np.ones((err_h, err_w), dtype=np.float64)
    cost_map = (grid.cost_left + grid.cost_top).astype(np.float64)
    cost_map /= cost_map.max() + EPS
    cost_region = fftconvolve(cost_map, region_mask, mode='valid')
    region_idx = np.argmax(cost_region)
    y1, x1 = np.unravel_index(region_idx, cost_region.shape)
    return Rect(x1, y1, x1 + err_w, y1 + err_h)


def get_sub_patch_offset(dst: np.ndarray, src: np.ndarray,
                         grid: PixelGrid) -> Point:
    src_h, src_w, _ = src.shape
    err_w = src_w // 4
    err_h = src_h // 4
    err_rect = get_error_region((err_w, err_h), grid)
    err_block = dst[err_rect.slice]
    src_mask = np.ones((src_h, src_w), dtype=np.bool)
    cost_map = get_ssd_cost_fft(src, src_mask, err_block, mode='valid')
    min_idx = np.argmin(cost_map)
    dy, dx = np.unravel_index(min_idx, cost_map.shape)
    offset = Point(err_rect.x1 - dx, err_rect.y1 - dy)
    return offset


def fill_texture(src: np.ndarray,
                 dst_size: SizeType,
                 placement: str,
                 refine_step: int,
                 update_seam: bool) -> np.ndarray:
    src_h, src_w, src_c = src.shape
    dst_c = src_c
    dst_w, dst_h = dst_size

    dst = np.zeros((dst_h, dst_w, dst_c), dtype=np.uint8)
    dst_mask = np.zeros((dst_h, dst_w), dtype=np.bool)
    bg = np.zeros((dst_h, dst_w, dst_c), dtype=np.uint8)
    bg_mask = np.zeros((dst_h, dst_w), dtype=np.bool)

    src_mask = np.ones((src_h, src_w), np.bool)

    grid = PixelGrid((dst_w, dst_h)) if update_seam else None

    while not dst_mask.all():
        offset = get_offset(dst, dst_mask, src, placement)
        print(offset)

        patch = paste(src, bg, offset)
        patch_mask = paste(src_mask, bg_mask, offset)

        blend(dst, dst_mask, patch, patch_mask, src, offset, grid)
        dst_mask |= patch_mask

    if update_seam:
        for _ in range(refine_step):
            offset = get_sub_patch_offset(dst, src, grid)

            patch = paste(src, bg, offset)
            patch_mask = paste(src_mask, bg_mask, offset)

            blend(dst, dst_mask, patch, patch_mask, src, offset, grid)

    return dst
