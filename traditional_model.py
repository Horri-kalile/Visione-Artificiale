import os
import cv2
import numpy as np
from features import get_combined_features
from sklearn.svm import SVC

def load_data_and_extract_features(data_dir):
    classes = sorted(os.listdir(data_dir))
    features_list = []
    labels_list = []

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                feat = get_combined_features(image)
                features_list.append(feat)
                labels_list.append(idx)

    return np.array(features_list), np.array(labels_list), classes

if __name__ == "__main__":
    print("Loading training data and extracting features...")
    X_train, y_train, classes = load_data_and_extract_features('data_split/train')
    print(f"Training set: {X_train.shape}")

    print("Training SVM classifier...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_train, y_train)