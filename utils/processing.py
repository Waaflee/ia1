import cv2
import numpy as np
from typing import Dict, Union, List
from matplotlib import pyplot as plt
from numpy.core.umath import log10
from math import copysign


def DEBUG(img, mode="cv", debug=False):
    if debug:
        if mode == "cv":
            cv2.imshow("debug", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            plt.imshow(img), plt.show()


def better_features(filename: str, debug=False) -> List:
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clean = cv2.GaussianBlur(gray, (3, 3), 0)
    _, clean = cv2.threshold(clean, 100, 255, 0)
    DEBUG(clean, debug)
    hm = cv2.HuMoments(cv2.moments(clean)).flatten()
    for i in range(0, 7):
        hm[i] = -1 * \
            copysign(1.0, hm[i]) * log10(abs(hm[i]))
    # return hm
    return [hm[0], hm[1], hm[3]]


def to_hu_moments(filename: str, debug=False) -> List[float]:
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
   # Binary Image
    ret, thresh = cv2.threshold(im, 127, 255, 0)
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

    DEBUG(im, mode="plt", debug=debug)

    # im = cv2.GaussianBlur(im, (5, 5), 0)
    _, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

    DEBUG(im, mode="plt", debug=debug)

    contours, hierarchy = cv2.findContours(
        im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours[1:]:
        cv2.fillPoly(im, pts=[cnt], color=(0, 0, 0))
    #     img = cv2.drawContours(im, cnt, -1, (255), cv2.FILLED)

    DEBUG(im, mode="plt", debug=debug)

    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * \
            copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments[:-3]
    # return huMoments


def to_ar_and_corners(filename: str, debug=False) -> Dict[str, Union[int, str, float]]:
    img = cv2.imread(filename)
    # Preprocessing
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # Binary Image
    ret, thresh = cv2.threshold(gray, 175, 255, 0)
    # Processing
    # Finding contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Sorting Contours
    # contours.sort(key=lambda x: len(x), reverse=True)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[1]  # First is a box containing the hole image.
    # ar, std = [], []
    # for cnt in contours[1:3]:
    # Find Corners
    # base = np.zeros(gray.shape, np.uint8)
    # # Draw only the countour
    # base_cnt = cv2.drawContours(base, cnt, -1, 255, 1)
    # # Convert to BGR
    # base_bgr = cv2.merge((base, base_cnt, base))
    # # Convert to Grayscale for corner detection
    # base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    # corners = cv2.goodFeaturesToTrack(base_gray, 25, 0.1, 10)
    # corners = np.int0(corners)

    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(base_gray, (x, y), 3, 255, -1)
    # DEBUG(base_gray, "plt", debug)

    # std_corner_deviation = corners.std()
    # std.append(corners.std())
    # Getting aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if aspect_ratio < 1:
        aspect_ratio = 1/aspect_ratio

        # ar.append(aspect_ratio)

        # aspect_ratio = int(round(aspect_ratio))

    return {
        "aspect_ratio": aspect_ratio,
        # "corners_deviation": std_corner_deviation
    }


def extract_features(filename: str, debug=False) -> List:
    # a = better_features(filename, debug)
    a = to_hu_moments(filename, debug)
    b = to_ar_and_corners(filename, debug)

    return np.array([a[0], a[1], a[3]] + [b["aspect_ratio"]], dtype=np.float).flatten()
    # return np.array( + [b["aspect_ratio"]], dtype=np.float).flatten()


def main() -> None:
    print("-"*25)
    print("Processing test: ")
    test = ["dataset/nails/nail", "dataset/screws/screw",
            "dataset/washers/washer", "dataset/nuts/nut"]
    test = ["dataset/nuts/nut"]
    postfix = ["_2.jpg", "_3.jpg", "_4.jpg"]
    # postfix = ["_1.jpg"]

    for i in test:
        for j in postfix:
            d = extract_features(f"{i}{j}", True)
            print(d)


if __name__ == "__main__":
    debug = True
    main()
