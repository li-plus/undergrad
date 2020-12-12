import cv2
import numpy as np
import argparse
import warp
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['perspective', 'sphere', 'sphere-inv'], default='perspective')
    args = parser.parse_args()

    if args.mode == 'perspective':
        src = cv2.imread('image/source.jpg')
        dst = cv2.imread('image/target.jpg')
        src_h, src_w, _ = src.shape
        src_points = np.array([(0, 0), (src_w, 0), (src_w, src_h), (0, src_h)], dtype=np.float32)
        dst_points = np.array([(192, 195), (536, 264), (508, 388), (169, 316)], dtype=np.float32)
        matrix = warp.get_perspective_transform(src_points, dst_points)
        mask = np.ones((src_h, src_w), dtype=np.float32)
        out = warp.warp_perspective(src, matrix, dst.shape)
        mask = warp.warp_perspective(mask, matrix, dst.shape[:2])
        out = (mask[:, :, None] < 0.5) * dst + out
        cv2.imwrite('output/warp_perspective.jpg', out)
    elif args.mode == 'sphere':
        src = cv2.imread('image/warping.png')
        dst_len = int(max(src.shape[:2]) / math.pi * 2)
        dst = warp.warp_sphere(src, (dst_len, dst_len))
        cv2.imwrite('output/warp_sphere.jpg', dst)
    else:
        src = cv2.imread('image/warping.png')
        dst_len = int(max(src.shape[:2]) / math.pi * 2)
        dst = warp.warp_sphere_inv(src, (dst_len, dst_len))
        cv2.imwrite('output/warp_sphere_inv.jpg', dst)


if __name__ == "__main__":
    main()
