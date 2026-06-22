import os
import cv2
import numpy as np
from features import get_combined_features
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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

    print("Loading validation data and extracting features...")
    X_val, y_val, _ = load_data_and_extract_features('data_split/val')
    print(f"Validation set: {X_val.shape}")

    print("Training SVM classifier...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))