import cv2


def display(name: str, img) -> None:
    # Resize for accurate output
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(0.1*width), int(0.1*height)),
                     interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img)
