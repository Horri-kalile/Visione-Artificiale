import os
import cv2
import numpy as np
import joblib
from features import get_combined_features

def load_traditional_model():
    clf = joblib.load('models/traditional_svm.joblib')
    classes = joblib.load('models/classes.joblib')
    return clf, classes

def evaluate_traditional(test_dir, clf, classes):
    X_test = []
    y_true = []

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                feat = get_combined_features(image)
                X_test.append(feat)
                y_true.append(idx)

    X_test = np.array(X_test)
    y_true = np.array(y_true)
    y_pred = clf.predict(X_test)
    return y_true, y_pred