import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def lut_brightness(delta):
    x = np.arange(256, dtype=np.int32)
    return np.clip(x + delta, 0, 255).astype(np.uint8)


def lut_contrast(alpha):
    x = np.arange(256, dtype=np.float32)
    return np.clip(127 + alpha * (x - 127), 0, 255).astype(np.uint8)


def lut_contrast_stretch(in_lo, in_hi, out_lo=0, out_hi=255):
    x = np.arange(256, dtype=np.float32)
    return np.clip(
        (out_hi - out_lo) / (in_hi - in_lo) * (x - in_lo) + out_lo, 0, 255
    ).astype(np.uint8)


def lut_gamma(gamma):
    x = np.arange(256, dtype=np.float32)
    return np.clip(255 * (x / 255) ** gamma, 0, 255).astype(np.uint8)


def lut_hist_eq(pdf):
    return (255 * np.cumsum(pdf)).astype(np.uint8)


def lut_hist_match(src_pdf, dst_pdf):
    src_eq = lut_hist_eq(src_pdf)
    dst_eq = lut_hist_eq(dst_pdf)

    dst_eq_inv = np.zeros(256, dtype=np.uint8)

    for i in range(255):
        y_curr = int(dst_eq[i])
        y_next = int(dst_eq[i + 1])
        if y_curr == y_next:
            dst_eq_inv[y_curr] = i
        else:
            y_mid = (y_curr + y_next) // 2
            dst_eq_inv[y_curr:y_mid] = i
            dst_eq_inv[y_mid:y_next] = i + 1

    dst_eq_inv[255] = 255
    lut = dst_eq_inv[src_eq]
    return lut


def get_pdf(image):
    values, counts = np.unique(image, return_counts=True)
    pdf = np.zeros(256, dtype=np.float32)
    pdf[values] = counts
    return pdf / pdf.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brightness', type=int, default=None)
    parser.add_argument('--contrast', type=float, default=None)
    parser.add_argument('--stretch', action='store_true')
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--hist-eq', action='store_true')
    parser.add_argument('--hist-match', action='store_true')
    parser.add_argument('--saturation', type=float, default=None)

    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='out.jpg')
    args = parser.parse_args()

    image = cv2.imread(args.source)
    target = cv2.imread(args.target)

    output = image
    if args.brightness:
        output = lut_brightness(args.brightness)[image]
    elif args.contrast:
        output = lut_contrast(args.contrast)[image]
    elif args.stretch:
        output = lut_contrast_stretch(image.min(), image.max())[image]
    elif args.gamma:
        output = lut_gamma(args.gamma)[image]
    elif args.hist_eq:
        output = np.concatenate([
            lut_hist_eq(get_pdf(image[:, :, 0]))[image[:, :, 0:1]],
            lut_hist_eq(get_pdf(image[:, :, 1]))[image[:, :, 1:2]],
            lut_hist_eq(get_pdf(image[:, :, 2]))[image[:, :, 2:3]],
        ], axis=-1)
    elif args.hist_match:
        output = np.concatenate([
            lut_hist_match(get_pdf(image[:, :, 0]), get_pdf(target[:, :, 0]))[image[:, :, 0:1]],
            lut_hist_match(get_pdf(image[:, :, 1]), get_pdf(target[:, :, 1]))[image[:, :, 1:2]],
            lut_hist_match(get_pdf(image[:, :, 2]), get_pdf(target[:, :, 2]))[image[:, :, 2:3]],
        ], axis=-1)
    elif args.saturation:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1].astype(np.float32) + args.saturation, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    cv2.imwrite(args.save_path, output)


if __name__ == "__main__":
    main()
