import cv2
import numpy as np


def extract_color_histogram(image, bins=(8, 8, 8)):
    # HSV separates chromatic information better than raw BGR for this task.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()

    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))
        else:
            hu_moments[i] = 0

    return hu_moments

def get_combined_features(image):
    color_feat = extract_color_histogram(image)
    shape_feat = extract_hu_moments(image)
    return np.hstack([color_feat, shape_feat])