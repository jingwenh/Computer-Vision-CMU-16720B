import numpy as np
import cv2
import BRIEF

import matplotlib.pyplot as plt

def rotate_match(img):
    """

    Args:
      img: numpy array representing the image with size (H, W, 3)
    Returns:
      match_count: Be in type of list.
                   Each item is a tuple (count of correct matches, degree)
    """
    h, w, _ = img.shape
    match_count = []
    for deg in range(0, 360, 10):
        M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1)
        img2 = cv2.warpAffine(img, M, (max(h, w), max(h, w)))

        locs1, desc1 = BRIEF.briefLite(img)
        locs2, desc2 = BRIEF.briefLite(img2)
        matches = BRIEF.briefMatch(desc1, desc2)
        BRIEF.plotMatches(img, img2, matches, locs1, locs2)

        match_count.append((matches.shape[0], deg))

    return match_count


def construct_bar_graph(match_count):
    """

    Args:
      match_count: Be in type of list.
                   Each item is a tuple (count of correct matches, degree)
    Returns:
    """
    count = [item[0] for item in match_count]
    degree = [item[1] for item in match_count]

    plt.bar(degree, count, width=5, align="center")
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../data/model_chickenbroth.jpg')
    match_count = rotate_match(img)
    construct_bar_graph(match_count)