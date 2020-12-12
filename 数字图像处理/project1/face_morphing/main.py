import numpy as np
import cv2
from scipy.spatial import Delaunay
import os
import json

import morph


def process(index):
    in_dir = 'inputs/{}'.format(index)
    out_dir = 'outputs/{}'.format(index)
    os.makedirs(out_dir, exist_ok=True)

    src = cv2.imread(os.path.join(in_dir, 'source.jpg'))
    dst = cv2.imread(os.path.join(in_dir, 'target.jpg'))

    src_points = json.load(open(os.path.join(in_dir, 'source.json')))
    dst_points = json.load(open(os.path.join(in_dir, 'target.json')))

    assert src.shape == dst.shape
    assert len(src_points) == len(dst_points)

    h, w, _ = src.shape

    src_points += [[0, 0], [0, h - 2], [w - 2, 0], [w - 2, h - 2]]
    dst_points += [[0, 0], [0, h - 2], [w - 2, 0], [w - 2, h - 2]]

    src_points = np.array(src_points, np.int32)
    dst_points = np.array(dst_points, np.int32)

    num_stages = 10
    for stage_index, ratio in enumerate(np.linspace(0, 1, num_stages)):
        if stage_index in [0, num_stages - 1]:
            continue

        out = morph.morph_image(src, src_points, dst, dst_points, ratio)
        out_path = os.path.join(out_dir, 'stage_{}.jpg'.format(stage_index))
        cv2.imwrite(out_path, out)


def main():
    for i in range(1, 3):
        process(i)


if __name__ == "__main__":
    main()
