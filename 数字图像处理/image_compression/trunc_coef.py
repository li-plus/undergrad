import numpy as np
import cv2
import argparse

import metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--keep-size', type=int, default=4)
    args = parser.parse_args()

    visualize = False

    image = cv2.imread('fig/lena.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    # example block
    x1, y1 = 262, 68
    x2, y2 = x1 + args.block_size, y1 + args.block_size
    block = image[y1:y2, x1:x2]

    dst_dct2d = np.zeros(image.shape, dtype=np.uint8)
    dst_dct1d1d = np.zeros(image.shape, dtype=np.uint8)

    for y1 in range(0, h, args.block_size):
        for x1 in range(0, w, args.block_size):
            x2, y2 = x1 + args.block_size, y1 + args.block_size
            block = image[y1:y2, x1:x2]

            # dct 2d
            # encode
            dct2d = cv2.dct(block.astype(np.float64))
            dct2d[args.keep_size:, :] = 0
            dct2d[:, args.keep_size:] = 0
            # decode
            dct2d_idct = cv2.idct(dct2d).astype(np.uint8)
            dst_dct2d[y1:y2, x1:x2] = dct2d_idct

            # first row then column
            # encode
            dct1d = cv2.dct(block.astype(np.float64), flags=cv2.DCT_ROWS)
            dct1d[:, args.keep_size:] = 0
            dct1d1d = cv2.dct(dct1d.T, flags=cv2.DCT_ROWS).T
            dct1d1d[args.keep_size:, :] = 0
            # decode
            dct1d1d_idct = cv2.idct(dct1d1d).astype(np.uint8)
            dst_dct1d1d[y1:y2, x1:x2] = dct1d1d_idct

            if visualize:
                cv2.imwrite('output/block.png', cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST))

                block_pos = cv2.rectangle(image.copy(), (x1-2, y1-2), (x2+2, y2+2), (0xff,0xff,0xff), 2)
                cv2.imwrite('output/block_pos.png', block_pos)

                dct2d = cv2.resize(dct2d, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('output/block_dct2d.png', dct2d)
                dct2d_idct = cv2.resize(dct2d_idct, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('output/block_dct2d_idct.png', dct2d_idct)

                dct1d1d = cv2.resize(dct1d1d, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('output/block_dct1d1d.png', dct1d1d)
                dct1d1d_idct = cv2.resize(dct1d1d_idct, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('output/block_dct1d1d_idct.png', dct1d1d_idct)

    print(f'dct2d PSNR:   {metric.get_psnr(dst_dct2d, image):.2f}')
    print(f'dct1d1d PSNR: {metric.get_psnr(dst_dct1d1d, image):.2f}')
    cv2.imwrite(f'output/dst_{args.block_size}_{args.keep_size}.png', dst_dct2d)


if __name__ == "__main__":
    main()
