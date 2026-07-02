import os
import cv2
import numpy as np
import joblib
from features import get_combined_features
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

def load_deep_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('models/deep_mobilenet.pth', map_location='cpu'))
    model.eval()
    return model

def evaluate_deep(test_dir, model, classes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                y_pred.append(pred.item())
                y_true.append(idx)

    return np.array(y_true), np.array(y_pred)