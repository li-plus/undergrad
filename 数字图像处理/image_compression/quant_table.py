import numpy as np
import cv2
import argparse

import metric


class QuantTable(object):
    @staticmethod
    def canon():
        return np.array([
            [1, 1, 1, 2, 3, 6, 8, 10],
            [1, 1, 2, 3, 4, 8, 9, 8],
            [2, 2, 2, 3, 6, 8, 10, 8],
            [2, 2, 3, 4, 7, 12, 11, 9],
            [3, 3, 8, 11, 10, 16, 15, 11],
            [3, 5, 8, 10, 12, 15, 16, 13],
            [7, 10, 11, 12, 15, 17, 17, 14],
            [14, 13, 13, 15, 15, 14, 14, 14]
        ], dtype=np.float64)

    @staticmethod
    def nikon():
        return np.array([
            [2, 1, 1, 2, 3, 5, 6, 7],
            [1, 1, 2, 2, 3, 7, 7, 7],
            [2, 2, 2, 3, 5, 7, 8, 7],
            [2, 2, 3, 3, 6, 10, 10, 7],
            [2, 3, 4, 7, 8, 13, 12, 9],
            [3, 4, 7, 8, 10, 12, 14, 11],
            [6, 8, 9, 10, 12, 15, 14, 12],
            [9, 11, 11, 12, 13, 12, 12, 12]
        ], dtype=np.float64)

    @staticmethod
    def jpeg():
        return np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=str, choices=['canon', 'nikon', 'jpeg'], default='jpeg')
    args = parser.parse_args()

    quant_table = {
        'canon': QuantTable.canon,
        'nikon': QuantTable.nikon,
        'jpeg': QuantTable.jpeg
    }[args.table]()

    block_size = 8

    image = cv2.imread('fig/lena.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    dst_dct2d = np.zeros(image.shape, dtype=np.uint8)
    num_compress = 0

    for y1 in range(0, h, block_size):
        for x1 in range(0, w, block_size):
            x2, y2 = x1 + block_size, y1 + block_size
            block = image[y1:y2, x1:x2].astype(np.float64) - 128
            # encode
            dct2d = cv2.dct(block)
            dct2d_q = np.round(dct2d / quant_table).astype(np.int32)
            # compress rate
            num_compress += (dct2d_q == 0).sum()
            # decode
            dct2d_idct = cv2.idct(dct2d_q * quant_table) + 128
            dst_dct2d[y1:y2, x1:x2] = dct2d_idct.astype(np.uint8)

    psnr = metric.get_psnr(dst_dct2d, image)
    print(f'table={args.table}, PSNR={psnr}, compress_rate={num_compress / image.size}')
    cv2.imwrite(f'output/quant_{args.table}.png', dst_dct2d)


if __name__ == "__main__":
    main()
