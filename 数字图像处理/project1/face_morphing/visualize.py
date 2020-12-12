import os
import numpy as np
import cv2
import json
from scipy.spatial import Delaunay


def process(src_path, landmarks_path, keypoints_save_path, delaunay_save_path):
    # draw landmarks
    src = cv2.imread(src_path)
    h, w, _ = src.shape

    points = json.load(open(landmarks_path))

    for point in points:
        src = cv2.circle(src, tuple(point), 3, (0xe0, 0xb2, 0), cv2.FILLED)

    cv2.imwrite(keypoints_save_path, src)

    # draw delaunay
    src = cv2.imread(src_path)

    points += [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    points = np.array(points, dtype=np.int32)

    delaunay = Delaunay(points)
    for tri in delaunay.simplices:
        p1, p2, p3 = points[tri]
        p1 = tuple(p1)
        p2 = tuple(p2)
        p3 = tuple(p3)
        src = cv2.line(src, p1, p2, (0x80, 0xff, 0), 1)
        src = cv2.line(src, p2, p3, (0x80, 0xff, 0), 1)
        src = cv2.line(src, p3, p1, (0x80, 0xff, 0), 1)

    for point in points:
        src = cv2.circle(src, tuple(point), 3, (0xe0, 0xb2, 0), cv2.FILLED)

    cv2.imwrite(delaunay_save_path, src)


def main():
    os.makedirs('outputs/1/', exist_ok=True)
    process('inputs/1/source.jpg', 'inputs/1/source.json', 'outputs/1/source_landmarks.jpg', 'outputs/1/source_delaunay.jpg')
    process('inputs/1/target.jpg', 'inputs/1/target.json', 'outputs/1/target_landmarks.jpg', 'outputs/1/target_delaunay.jpg')
    os.makedirs('outputs/2/', exist_ok=True)
    process('inputs/2/source.jpg', 'inputs/2/source.json', 'outputs/2/source_landmarks.jpg', 'outputs/2/source_delaunay.jpg')
    process('inputs/2/target.jpg', 'inputs/2/target.json', 'outputs/2/target_landmarks.jpg', 'outputs/2/target_delaunay.jpg')


if __name__ == "__main__":
    main()
