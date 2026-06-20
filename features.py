import cv2


def extract_color_histogram(image, bins=(8, 8, 8)):
    # HSV separates chromatic information better than raw BGR for this task.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()