import argparse
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


def get_features(image: np.ndarray,
                 method: str
                 ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    else:
        raise ValueError(f'Unsupported feature extractor {method}')

    # Get key points and feature descriptors
    key_points, features = descriptor.detectAndCompute(image, mask=None)
    return key_points, features


def get_matcher(method: str, cross_check: bool) -> cv2.BFMatcher:
    if method == 'sift' or method == 'surf':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    elif method == 'orb' or method == 'brisk':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    else:
        raise ValueError(f'Invalid method {method}')

    return matcher


def get_bf_matches(src_feat: np.ndarray,
                   dst_feat: np.ndarray,
                   method: str
                   ) -> List[cv2.DMatch]:
    matcher = get_matcher(method, cross_check=True)

    # Match descriptors
    best_matches = matcher.match(src_feat, dst_feat)

    # Sort the matches in order of distance
    raw_matches = sorted(best_matches, key=lambda x: x.distance)
    print('Raw matches (Brute force):', len(raw_matches))
    return raw_matches


def get_knn_matches(src_feat: np.ndarray,
                    dst_feat: np.ndarray,
                    ratio: float,
                    method: str
                    ) -> List[cv2.DMatch]:
    matcher = get_matcher(method, cross_check=False)
    # compute the raw matches and initialize the list of actual matches
    raw_matches = matcher.knnMatch(src_feat, dst_feat, k=2)
    print('Raw matches (knn):', len(raw_matches))
    matches = []

    # loop over the raw matches
    for m, n in raw_matches:
        # ensure the distance is within a certain ratio of each other
        # (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def get_homography(src: np.ndarray,
                   dst: np.ndarray,
                   feature_extractor: str,
                   feature_matcher: str
                   ) -> np.ndarray:
    # Find the homography matrix to warp src image to dst viewpoint
    src_pts, src_feat = get_features(src, feature_extractor)
    dst_pts, dst_feat = get_features(dst, feature_extractor)

    cv2.imwrite('pts.jpg', np.hstack((
        cv2.drawKeypoints(src, src_pts, None, color=(0, 255, 0)),
        cv2.drawKeypoints(dst, dst_pts, None, color=(0, 255, 0)),
    )))

    if feature_matcher == 'bf':
        matches = get_bf_matches(src_feat, dst_feat, feature_extractor)
    elif feature_matcher == 'knn':
        matches = get_knn_matches(src_feat, dst_feat, 0.75, feature_extractor)
    else:
        raise ValueError(f'Unsupported feature matcher {feature_matcher}')

    cv2.imwrite('matches.jpg', cv2.drawMatches(
        src, src_pts, dst, dst_pts, matches[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    src_pts = np.float32([src_pts[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([dst_pts[m.trainIdx].pt for m in matches])

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                   ransacReprojThreshold=4)
    if H is None:
        raise RuntimeError('Failed to find homography matrix')
    return H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='sift',
                        choices=['sift', 'surf', 'brisk', 'orb'])
    parser.add_argument('--matcher', type=str, default='bf',
                        choices=['bf', 'knn'])
    args = parser.parse_args()

    # for campus
    # in_dir = '../data/campus'
    # out_dir = '../output/campus_align'
    # scale = 0.5
    # align_w = 785
    # align_h = 341
    # warp_to = 0
    # pad_left = 0

    # for golden gate
    in_dir = '../data/goldengate'
    out_dir = '../output/golden_align'
    scale = 0.5
    align_w = 1100
    align_h = 450
    warp_to = 3
    pad_left = 500

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read images
    imgs = []
    masks = []
    in_paths = sorted(Path(in_dir).glob('*.jpg'))
    for path in in_paths:
        img = cv2.imread(str(path))
        assert img is not None
        img = cv2.resize(img, None, fx=scale, fy=scale)
        mask = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.pad(img, ((0, 0), (pad_left, 0), (0, 0)))
        mask = np.pad(mask, ((0, 0), (pad_left, 0)))
        imgs.append(img)
        masks.append(mask)

    # Compute homography for each image
    Hs = [np.eye(3, dtype=np.float32)]
    for i in range(len(imgs) - 1):
        dst = imgs[i]
        src = imgs[i + 1]
        prev_H = Hs[-1]
        H = get_homography(src, dst, args.feature, args.matcher)
        Hs.append(prev_H @ H)
        print(H)

    to_H_inv = np.linalg.inv(Hs[warp_to])
    Hs = [H @ to_H_inv for H in Hs]

    # Warp and save
    for src, src_mask, H, src_in in zip(imgs, masks, Hs, in_paths):
        src_mask_align = cv2.warpPerspective(src_mask, H, (align_w, align_h))
        src_align = cv2.warpPerspective(src, H, (align_w, align_h))

        assert cv2.imwrite(str(out_dir / f'{src_in.stem}.jpg'), src_align)
        assert cv2.imwrite(str(out_dir / f'{src_in.stem}.png'), src_mask_align)


if __name__ == '__main__':
    main()
