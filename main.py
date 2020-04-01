import cv2
import numpy as np
import os


def print_img(name, img):
    # Resize for accurate output
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(0.1*width), int(0.1*height)),
                     interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img)


print("-"*25)
types = ["nail", "screw", "washer", "nut"]
types = ["rotated"]
types = os.listdir("dataset/nails")
types = ["dataset/nails/" + i for i in types]
# nails = os.listdir("dataset/nails")
# screws = os.listdir("dataset/screws")
# washers = os.listdir("dataset/washers")
# nuts = os.listdir("dataset/nuts")

# dataset = [os.listdir(f"dataset/{i}s") for i in types]


for i in types:
    # filename = f'{i}_1.jpg'
    filename = i
    print(filename)
    # filename = f'washer_1.jpg'
    img = cv2.imread(filename)
    # Preprocessing
    # Denoise
    img = cv2.blur(img, (3, 3))
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
    # dst = cv2.dilate(dst, None)
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
    # corners = 0
    ar = []
    for cnt in contours[1:4]:
        # Count corners over the countour
        # mask = np.zeros(gray.shape, np.uint8)
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
        ar.append(aspect_ratio)

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x), int(y))
        # radius = int(radius)
        # mask = np.zeros(gray.shape, np.uint8)
        # outer_circle = cv2.circle(mask, center, radius, (255, 255, 255), 2)
        # outer_countour = cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)
        # # outer_circle = cv2.cvtColor(outer_circle, cv2.COLOR_BGR2GRAY)
        # outer_countour = cv2.cvtColor(outer_countour, cv2.COLOR_BGR2GRAY)
        # # ret, outer_circle = cv2.threshold(outer_circle, 250, 255, 0)
        # ret, outer_countour = cv2.threshold(outer_countour, 250, 255, 0)

        # matching_circle = np.logical_and(outer_circle, outer_countour)
        # unique, counts = np.unique(matching_circle, return_counts=True)
        # print("DIIIICCCTT", dict(zip(unique, counts)))
        # img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
    try:
        corners = corner_data[255]
    except KeyError as e:
        corners = 0
    # Extracted features
    print("File: ", filename)
    print("Aspect Ratios: ", ar)
    # print("Corners: ", corner_data)
    print("Corners: ", corners)
    print()
    # if ar.count(0) >= 2:
    #     print("\tCircular")
    # if ar.count(1) >= 2:
    #     print("\trectangular")

    # Post Processing
    # Resize for accurate output
    # height, width = img.shape[:2]
    # img = cv2.resize(img, (int(0.1*width), int(0.1*height)),
    #                  interpolation=cv2.INTER_AREA)
    # cv2.imshow(filename, img)
    # cv2.waitKey(0)
