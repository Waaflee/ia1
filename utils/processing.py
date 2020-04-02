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
    # im = cv2.imread(filename)
    # Preprocessing
    # Grayscale
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
   # Binary Image
    ret, thresh = cv2.threshold(im, 100, 255, 0)
    # Processing
    # Finding contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Sorting Contours
    # contours.sort(key=lambda x: len(x), reverse=True)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[1]  # First is a box containing the hole image.

    x, y, w, h = cv2.boundingRect(cnt)
    im = im[y:y+h, x:x+w]

    if debug:
        cv2.imshow("binary", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    im = cv2.GaussianBlur(im, (5, 5), 0)
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


def to_ar_and_corners(filename: str) -> Dict[str, Union[int, str, float]]:
    img = cv2.imread(filename)
    # Preprocessing
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # Binary Image
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    # Processing
    # Finding contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Sorting Contours
    # contours.sort(key=lambda x: len(x), reverse=True)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    cnt = contours[1]  # First is a box containing the hole image.

    # Find Corners
    base = np.zeros(gray.shape, np.uint8)
    # Draw only the countour
    base_cnt = cv2.drawContours(base, cnt, -1, 100, 1)
    # Convert to BGR
    base_bgr = cv2.merge((base, base_cnt, base))
    # Convert to Grayscale for corner detection
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(base_gray, 5, 0.5, 10)
    corners = np.int0(corners)

    # print("Mean: ", corners.mean())
    # print("Deviation: ", corners.std())

    if debug:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(base_gray, (x, y), 3, 255, -1)

        plt.imshow(base_gray), plt.show()

    corners = np.sort(corners)
    corners_a = corners[:int(len(corners)/2)]
    corners_b = corners[int(len(corners)/2):]
    mean_a = corners_a.mean()
    mean_b = corners_b.mean()

    # corners = min([abs(mean_a - mean_b), corners.std()])
    # corners = corners.std()
    corners = abs(mean_a - mean_b)

    # # Denoise a little bit
    # # base_gray = cv2.blur(base_gray, (10, 10))
    # base_gray = cv2.GaussianBlur(base_gray, (9, 9), 0)
    # # To float for harris detection algorithm
    # base_gray = np.float32(base_gray)
    # # Harris corner detections, adjustable parameters
    # dst = cv2.cornerHarris(base_gray, 11, 5, 0.04)
    # # Dilating corners, non important
    # # dst = cv2.dilate(dst, None)

    # # Threshold to mark corners, adjustable
    # base_bgr[dst > 0.05*dst.max()] = [0, 0, 255]

    # # Counting corners of figure/countour
    # unique, counts = np.unique(base_bgr, return_counts=True)
    # base_bgr_data = dict(zip(unique, counts))

    # # Amount of corners detected and filtered
    # corners = int(base_bgr_data[255])

    # Getting aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if aspect_ratio < 1:
        aspect_ratio = 1/aspect_ratio

    # aspect_ratio = int(round(aspect_ratio))

    # if debug:
    #     # display(filename, base_bgr)
    #     return {
    #         "filename": filename,
    #         "aspect_ratio": aspect_ratio,
    #         "corners": corners
    #     }, base_bgr

    return {
        "filename": filename,
        "aspect_ratio": aspect_ratio,
        "corners": corners
    }


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
