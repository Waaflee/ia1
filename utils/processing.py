import cv2
import numpy as np
from typing import Dict, Union


def extract_features(filename: str) -> Dict[str, Union[int, str, float]]:
    # filename = i
    # filename = f'washer_1.jpg'
    img = cv2.imread(filename)
    # Preprocessing
    # Denoise
    # img = cv2.blur(img, (3, 3))
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("res", res)
    # Binary Image
    ret, thresh = cv2.threshold(gray, 100, 255, 0)

    # Processing
    # Finding contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Sorting Contours
    # contours.sort(key=lambda x: len(x), reverse=True)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    corners = img.copy()
    # corners[dst > 0.01*dst.max()] = [255, 255, 255]
    corners[dst > 0.01*dst.max()] = [0, 0, 255]
    bin_corners = cv2.cvtColor(corners, cv2.COLOR_BGR2GRAY)
    # print_img("", corners)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ret, bin_corners = cv2.threshold(bin_corners, 250, 255, 0)
    unique, counts = np.unique(bin_corners, return_counts=True)
    corner_data = dict(zip(unique, counts))
    # continue
    # Threshold for an optimal value, it may vary depending on the image.
    corners = 0
    ar = 0
    cnt = contours[1]
    # epsilon = 10**-10*cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)

    base = np.zeros(gray.shape, np.uint8)
    # base.fill(100)
    base_cnt = cv2.drawContours(base, cnt, -1, 250, 1)
    base_bgr = cv2.merge((base, base_cnt, base))
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.blur(base_gray, (5, 5))
    base_gray = np.float32(base_gray)
    dst = cv2.cornerHarris(base_gray, 10, 21, 0.01)
    dst = cv2.dilate(dst, None)
    base_bgr[dst > 0.1*dst.max()] = [0, 0, 255]

    # bin_corners = cv2.cvtColor(corners, cv2.COLOR_BGR2GRAY)
    # print_img("", corners)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # ret, bin_corners = cv2.threshold(bin_corners, 250, 255, 0)
    unique, counts = np.unique(base_bgr, return_counts=True)
    base_bgr_data = dict(zip(unique, counts))

    corners = int(base_bgr_data[255])

    # Count corners over the countour
    # mask = cv2.drawContours(mask, [cnt], 0, 255, 5)
    # corners_on_cnt = np.logical_and(bin_corners, mask)
    # unique, counts = np.unique(corners_on_cnt, return_counts=True)
    # corners_dict = dict(zip(unique, counts))
    # print(corners_dict)
    # try:
    #     corners += corners_dict[True]
    # except KeyError as e:
    #     # print("No matches")
    #     pass

    # print_img("mask", mask)
    # # print_img("bin_corbin_corners", bin_corners)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    ar = aspect_ratio
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # center = (int(x), int(y))
    # radius = int(radius)
    # mask = np.zeros(gray.shape, np.uint8)
    # img = cv2.circle(img, center, radius, (255, 255, 255), 2)
    # outer_countour = cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)
    # outer_circle = cv2.cvtColor(outer_circle, cv2.COLOR_BGR2GRAY)
    # outer_countour = cv2.cvtColor(outer_countour, cv2.COLOR_BGR2GRAY)
    # ret, outer_circle = cv2.threshold(outer_circle, 250, 255, 0)
    # ret, outer_countour = cv2.threshold(outer_countour, 250, 255, 0)

    # matching_circle = np.logical_and(outer_circle, outer_countour)
    # unique, counts = np.unique(matching_circle, return_counts=True)
    # print("Circle Matching", dict(zip(unique, counts)))
    # img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

    # # Extracted features
    # print("File: ", filename)
    # print("Aspect Ratios: ", ar)
    # # print("Corners: ", corner_data)
    # print("Corners: ", corners)
    # print()
    return {
        "filename": filename,
        "aspect_ratio": ar,
        "corners": corners
    }
