import math
import numpy as np
import cv2
import json
import os
from scipy.spatial import Delaunay
import drawer
import prewarp
import morph_view
from face_morphing import morph


def load_points(path):
    points = json.load(open(path))
    points = np.array(points, dtype=np.float32)
    return points


def process(index):
    in_dir = 'inputs/{}'.format(index)
    out_dir = 'outputs/{}'.format(index)
    os.makedirs(out_dir, exist_ok=True)

    src = cv2.imread(os.path.join(in_dir, 'source.jpg'))
    dst = cv2.imread(os.path.join(in_dir, 'target.jpg'))
    src_points = load_points(os.path.join(in_dir, 'source_dlib.json'))
    dst_points = load_points(os.path.join(in_dir, 'target_dlib.json'))

    assert src.shape == dst.shape
    assert src_points.shape == dst_points.shape

    F, _ = cv2.findFundamentalMat(src_points, dst_points, cv2.FM_8POINT)
    F = F.astype(np.float32)
    H0, H1 = prewarp.get_prewarp(F)

    src_points_extra = load_points(os.path.join(in_dir, 'source_hand.json'))
    if len(src_points_extra):
        src_points = np.vstack((src_points, src_points_extra))

    dst_points_extra = load_points(os.path.join(in_dir, 'target_hand.json'))
    if len(dst_points_extra):
        dst_points = np.vstack((dst_points, dst_points_extra))

    h, w = src.shape[:2]
    corners = np.array([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)], dtype=np.float32)
    src_points = np.vstack((src_points, corners)).astype(np.float32)
    dst_points = np.vstack((dst_points, corners)).astype(np.float32)

    num_stages = 5
    for stage_index, ratio in enumerate(np.linspace(0, 1, num_stages)):
        if stage_index in [0, num_stages - 1]:
            continue

        merged = morph_view.morph_view(src, src_points, dst, dst_points, ratio, H0, H1)
        cv2.imwrite(os.path.join(out_dir, 'stage_{}.jpg'.format(stage_index)), merged)


def main():
    for i in range(1, 5):
        process(i)


if __name__ == "__main__":
    main()
