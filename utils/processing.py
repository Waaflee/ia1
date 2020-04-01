import cv2
import numpy as np
from typing import Dict, Union


def extract_features(filename: str) -> Dict[str, Union[int, str, float]]:
    img = cv2.imread(filename)
    # Preprocessing

    # Denoise
    # img = cv2.blur(img, (3, 3))

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
    base_cnt = cv2.drawContours(base, cnt, -1, 250, 1)
    # Convert to BGR
    base_bgr = cv2.merge((base, base_cnt, base))
    # Convert to Grayscale for corner detection
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    # Denoise a little bit
    base_gray = cv2.blur(base_gray, (5, 5))
    # To float for harris detection algorithm
    base_gray = np.float32(base_gray)
    # Harris corner detections, adjustable parameters
    dst = cv2.cornerHarris(base_gray, 10, 21, 0.01)
    # Dilating corners, non important
    dst = cv2.dilate(dst, None)

    # Threshold to mark corners, adjustable
    base_bgr[dst > 0.1*dst.max()] = [0, 0, 255]

    # Counting corners of figure/countour
    unique, counts = np.unique(base_bgr, return_counts=True)
    base_bgr_data = dict(zip(unique, counts))

    # Amount of corners detected and filtered
    corners = int(base_bgr_data[255])

    # Getting aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h

    return {
        "filename": filename,
        "aspect_ratio": aspect_ratio,
        "corners": corners
    }
