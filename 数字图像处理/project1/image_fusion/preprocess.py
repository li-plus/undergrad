import cv2
import numpy as np
import os


def translate(src, dx, dy, dst_shape):
    dst_h, dst_w, _ = dst_shape
    src_h, src_w, _ = src.shape
    dst = np.zeros(dst_shape, np.uint8)
    for dst_r in range(dst_h):
        for dst_c in range(dst_w):
            src_r = dst_r - dy
            src_c = dst_c - dx
            if -1 < src_r < src_h and -1 < src_c < src_w:
                dst[dst_r, dst_c] = src[src_r, src_c]
    return dst


def process(in_src, in_mask, in_dst, dx, dy, out_dir):
    mask = cv2.imread(in_mask, cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(in_src)
    dst = cv2.imread(in_dst)

    mask = (mask > 127).astype(np.uint8)
    mask = mask[:, :, None]
    h, w, _ = dst.shape
    mask = translate(mask, dx, dy, (h, w, 1))
    src = translate(src, dx, dy, (h, w, 3))
    
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'mask.png'), mask * 255)
    cv2.imwrite(os.path.join(out_dir, 'source.jpg'), src)
    cv2.imwrite(os.path.join(out_dir, 'target.jpg'), dst)


def main():
    os.makedirs('inputs/1', exist_ok=True)
    process('fig/test1_src.jpg', 'fig/test1_mask.jpg', 'fig/test1_target.jpg', -82, -90, 'inputs/1/')
    os.makedirs('inputs/2', exist_ok=True)
    process('fig/test2_src.png', 'fig/test2_mask.png', 'fig/test2_target.png', 144, 162, 'inputs/2/')


if __name__ == "__main__":
    main()
