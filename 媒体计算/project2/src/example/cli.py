import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import seam_carving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('-o', dest='dst', type=str, required=True)
    parser.add_argument('--keep', type=str, default=None)
    parser.add_argument('--drop', type=str, default=None)
    parser.add_argument('--dw', type=int, default=0)
    parser.add_argument('--dh', type=int, default=0)
    parser.add_argument('--energy', type=str, default='backward',
                        choices=['backward', 'forward'])
    parser.add_argument('--order', type=str, default='width-first',
                        choices=['width-first', 'height-first', 'optimal'])
    parser.add_argument('--face', action='store_true')
    args = parser.parse_args()

    try:
        print('Loading source image from {}'.format(args.src))
        src = np.array(Image.open(args.src))

        drop_mask = None
        if args.drop is not None:
            print('Loading drop_mask from {}'.format(args.drop))
            drop_mask = np.array(Image.open(args.drop).convert('L'))

        keep_mask = None
        if args.keep is not None:
            print('Loading keep_mask from {}'.format(args.keep))
            keep_mask = np.array(Image.open(args.keep).convert('L'))

        if args.face:
            # Face detection using face++ API
            with open(Path(__file__).parent / 'api_config.json') as f:
                api_config = json.load(f)
            with open(args.src, 'rb') as f:
                img_data = f.read()
            image_base64 = base64.b64encode(img_data)
            url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
            data = {
                'api_key': api_config['api_key'],
                'api_secret': api_config['api_secret'],
                'image_base64': image_base64
            }
            response = requests.post(url, data)
            data = response.json()

            src_h, src_w, _ = src.shape
            if keep_mask is None:
                keep_mask = np.zeros((src_h, src_w), dtype=np.bool)

            for face in data['faces']:
                rect = face['face_rectangle']
                x1 = rect['left']
                y1 = rect['top']
                w = rect['width']
                h = rect['height']
                keep_mask[y1:y1 + h, x1:x1 + w] = True

        print('Performing seam carving...')
        start = time.time()
        if drop_mask is not None:
            dst = seam_carving.remove_object(src, drop_mask, keep_mask)
        else:
            src_h, src_w, _ = src.shape
            dst = seam_carving.resize(src, (src_w + args.dw, src_h + args.dh),
                                      args.energy, args.order, keep_mask)
        print('Done at {:.4f} second(s)'.format(time.time() - start))

        print('Saving output image to {}'.format(args.dst))
        Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(dst).save(args.dst)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
