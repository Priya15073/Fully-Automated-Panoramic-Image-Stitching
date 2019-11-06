import cv2
from imutils import paths
import numpy as np
import argparse
import imutils

def stitch(image1, image2, lowe_ratio=0.75, max_t=4.0, is_match=False):
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    descriptors = cv2.xfeatures2d.SIFT_create()
    (key_points_2, features_2) = descriptors.detectAndCompute(image2, None)
    key_points_2 = np.float32([i.pt for i in key_points_2])

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    descriptors = cv2.xfeatures2d.SIFT_create()
    (key_points_1, features_1) = descriptors.detectAndCompute(image1, None)
    key_points_1 = np.float32([i.pt for i in key_points_1])

    match_instance = cv2.DescriptorMatcher_create("BruteForce")
    all_matches = match_instance.knnMatch(features_2, features_1, 2)

    matches = []

    for val in all_matches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            matches.append((val[0].trainIdx, val[0].queryIdx))

    if len(matches) < 5:
        return None

    points2 = np.float32([key_points_2[i] for (_,i) in matches])
    points1 = np.float32([key_points_1[i] for (i,_) in matches])

    (Homography, status) = cv2.findHomography(points2, points1, cv2.RANSAC, max_t)

    val = image2.shape[1] + image1.shape[1]
    result = cv2.warpPerspective(image2, Homography, (val , image2.shape[0]))
    result[0:image1.shape[0], 0:image1.shape[1]] = image1

    if is_match:
        (h2,w2) = image2.shape[:2]
        (h1, w1) = image1.shape[:2]
        vis = np.zeros((max(h2, h1), w2 + w1, 3), dtype="uint8")
        vis[0:h2, 0:w2] = image2
        vis[0:h1, w2:] = image1

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(key_points_2[queryIdx][0]), int(key_points_2[queryIdx][1]))
                ptB = (int(key_points_1[trainIdx][0]) + w2, int(key_points_1[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return (result, vis)

    return result


def get_input(image_args):
    filename = []
    image_paths = sorted(list(paths.list_images(image_args)))
    images = []

    for path in image_paths:
        x = cv2.imread(path)
        x = imutils.resize(x,width=400)
        x = imutils.resize(x,height=400)
        images.append(x)
    return images


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--images", type=str, required=True)
    arg.add_argument("-o", "--output", type=str, required=True)
    args = vars(arg.parse_args())

    images = get_input(args["images"])

    if(len(images)) == 2:
        (result, Z) = stitch(images[0], images[1], is_match=True)
    else:
        (result, Z) = stitch(images[-2], images[-1], is_match=True)
        for i in range(len(images) - 2):
            (result, Z) = stitch(images[len(images)-i-3], result, is_match=True)

    cv2.imwrite(args["output"], result)
