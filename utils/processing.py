import cv2
import numpy as np
from typing import Dict, Union, List
from matplotlib import pyplot as plt
from numpy.core.umath import log10
from math import copysign


def display(name: str, img) -> None:
    # Resize for accurate output
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(0.1*width), int(0.1*height)),
                     interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img)


def to_hu_moments(filename: str) -> List[float]:
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
   # Binary Image
    ret, thresh = cv2.threshold(im, 100, 255, 0)
    # Processing
    # Finding contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Sorting Contours
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[1]  # First is a box containing the hole image.

    x, y, w, h = cv2.boundingRect(cnt)
    # Cutted image to avoid shadows.
    im = im[y:y+h, x:x+w]

    if debug:
        cv2.imshow("binary", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # im = cv2.GaussianBlur(im, (5, 5), 0)
    _, im = cv2.threshold(im, 95, 255, cv2.THRESH_BINARY)

    if debug:
        cv2.imshow("binary", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * \
            copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments
    # return huMoments


debug = False


def main() -> None:
    print("-"*25)
    print("Processing test: ")
    test = ["dataset/nails/nail", "dataset/screws/screw",
            "dataset/washers/washer", "dataset/nuts/nut"]
    postfix = ["_1.jpg", "_2.jpg", "_3.jpg"]
    postfix = ["_1.jpg"]
    data = []
    for i in test:
        for j in postfix:
            d = to_hu_moments(f"{i}{j}")
            data.append(d)
            print(d)


if __name__ == "__main__":
    debug = True
    main()
