import json
import argparse
import dlib
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--save-path', type=str, default='landmarks.json')
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    src = cv2.imread(args.src)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    rects = detector(src, 0)

    assert len(rects) == 1

    shape = predictor(src, rects[0])

    coords = []

    for i in range(shape.num_parts):
        point = (shape.part(i).x, shape.part(i).y)
        coords.append(point)

    with open(args.save_path, 'w') as f:
        json.dump(coords, f)


if __name__ == "__main__":
    main()
