from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import graphcut


def main():
    parser = argparse.ArgumentParser('GraphCut-Texture')
    parser.add_argument('img', type=str, help='Input texture')
    parser.add_argument('--placement', type=str, default='entire-patch',
                        choices=['random', 'entire-patch'],
                        help='Patch placement policy')
    parser.add_argument('--out', type=str, default='../output/',
                        help='Output folder')
    parser.add_argument('--refine', type=int, default=5,
                        help='Steps to refine using sub-patch matching')
    parser.add_argument('--sx', type=float, default=2.,
                        help='Scale to expand along x-axis')
    parser.add_argument('--sy', type=float, default=2.,
                        help='Scale to expand along y-axis')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--no-oldcut', action='store_false', dest='oldcut',
                        help='Disable old cut update')
    args = parser.parse_args()

    assert args.sx >= 1 and args.sy >= 1

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load image
    img_path = Path(args.img)
    src = np.asarray(Image.open(img_path).convert('RGB'))

    # create output folder
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (img_path.stem + '.png')

    # calculate output shape
    src_h, src_w, _ = src.shape
    dst_h = int(args.sy * src_h)
    dst_w = int(args.sx * src_w)

    # fill texture
    dst = graphcut.fill_texture(
        src, (dst_w, dst_h), args.placement, args.refine, args.oldcut)

    # save output
    Image.fromarray(dst).save(out_path)


if __name__ == '__main__':
    main()
