import cv2
import argparse
import seam_carving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--delta-col', type=int, default=0)
    args = parser.parse_args()

    src = cv2.imread(args.src)

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        dst = seam_carving.remove_object(src, mask)
    else:
        if args.delta_col > 0:
            dst = seam_carving.insert_seams(src, args.delta_col)
        else:
            dst = seam_carving.remove_seams(src, abs(args.delta_col))

    cv2.imwrite(args.dst, dst)


if __name__ == "__main__":
    main()
