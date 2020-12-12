import cv2
import numpy as np
import os
import json
from face_morphing import preprocess


def process(index):
    out_dir = os.path.join('inputs', str(index))
    os.makedirs(out_dir, exist_ok=True)

    src_basename = 'source_{}'.format(index)
    dst_basename = 'target_{}'.format(index)
    src_path = None
    dst_path = None

    fig_dir = 'fig'
    for filename in os.listdir(fig_dir):
        if filename.startswith(src_basename):
            src_path = os.path.join(fig_dir, filename)
        elif filename.startswith(dst_basename):
            dst_path = os.path.join(fig_dir, filename)

    assert src_path is not None
    assert dst_path is not None

    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    h, w = dst.shape[:2]
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(out_dir, 'source.jpg'), src)
    cv2.imwrite(os.path.join(out_dir, 'target.jpg'), dst)

    label_path = os.path.join('landmarks', str(index))
    preprocess.convert_landmarks(os.path.join(label_path, 'source_hand.txt'), os.path.join(out_dir, 'source_hand.json'))
    preprocess.convert_landmarks(os.path.join(label_path, 'target_hand.txt'), os.path.join(out_dir, 'target_hand.json'))
    preprocess.convert_landmarks(os.path.join(label_path, 'source_dlib.txt'), os.path.join(out_dir, 'source_dlib.json'))
    preprocess.convert_landmarks(os.path.join(label_path, 'target_dlib.txt'), os.path.join(out_dir, 'target_dlib.json'))


def main():
    for i in range(1, 5):
        process(i)


if __name__ == "__main__":
    main()
