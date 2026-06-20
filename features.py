import cv2


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