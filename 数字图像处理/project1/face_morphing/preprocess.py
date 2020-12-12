import os
import cv2
import json


def convert_landmarks(in_path, out_path):
    points = []

    with open(in_path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x, y = line.strip().split(' ')
            points.append((int(x), int(y)))

    with open(out_path, 'w') as f:
        json.dump(points, f)


def process(index):
    os.makedirs('inputs/{}'.format(index), exist_ok=True)
    source = cv2.imread('fig/source{}.png'.format(index))
    target = cv2.imread('fig/target{}.png'.format(index))
    h, w, _ = target.shape
    source = cv2.resize(source, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('inputs/{}/source.jpg'.format(index), source)
    cv2.imwrite('inputs/{}/target.jpg'.format(index), target)

    convert_landmarks('landmarks/{}/source.txt'.format(index), 'inputs/{}/source.json'.format(index))
    convert_landmarks('landmarks/{}/target.txt'.format(index), 'inputs/{}/target.json'.format(index))


def main():
    for i in range(1, 3):
        process(i)


if __name__ == "__main__":
    main()
