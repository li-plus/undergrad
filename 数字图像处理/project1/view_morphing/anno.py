import cv2
import argparse


def load_landmarks(in_path):
    points = []

    with open(in_path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x, y = line.strip().split(' ')
            points.append((int(x), int(y)))

    return points


point = (-1, -1)
points = []

def get_coords(event, x, y, flags, param):
    global point, points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print('Add point {}: {}'.format(len(points), point))


def main():
    global point, points
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--anno', type=str, required=True)
    args = parser.parse_args()

    src_bak = cv2.imread(args.src)
    src = src_bak.copy()

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img', get_coords)

    point_click = (-1, -1)
    points = load_landmarks(args.anno)

    while True:
        cv2.imshow('img', src)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('q'):
            print("--------------- Final Points -------------------")
            for x, y in points:
                print('{} {}'.format(x, y))
            break
        elif k == ord('z'):
            if points:
                revert = points.pop()
                print('Revert point {}: {}'.format(len(points), revert))

        if point_click != point:
            point_click = point
            points.append(point_click)

        src = src_bak.copy()
        for p in points:
            cv2.circle(src, p, 3, (0, 255, 0), cv2.FILLED)


if __name__ == "__main__":
    main()
