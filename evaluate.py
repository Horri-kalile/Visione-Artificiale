import os
import cv2
import numpy as np
import joblib
from features import get_combined_features
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_traditional_model():
    clf = joblib.load('models/traditional_svm.joblib')
    classes = joblib.load('models/classes.joblib')
    scaler = joblib.load('models/scaler.joblib')  
    return clf, classes, scaler  

def evaluate_traditional(test_dir, clf, classes, scaler):  
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
    
    
    X_test = scaler.transform(X_test)
    
    
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

if __name__ == "__main__":
    test_dir = 'data_split/test'
    clf, classes, scaler = load_traditional_model()  
    deep_model = load_deep_model(len(classes))
    
    print("Evaluating Traditional Approach...")
    y_true_trad, y_pred_trad = evaluate_traditional(test_dir, clf, classes, scaler)  
    
    print("Evaluating Deep Learning Approach...")
    y_true_deep, y_pred_deep = evaluate_deep(test_dir, deep_model, classes)
    
    # Print basic results to console
    print("\n--- TRADITIONAL RESULTS ---")
    print(f"Accuracy: {accuracy_score(y_true_trad, y_pred_trad):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true_trad, y_pred_trad))
    
    print("\n--- DEEP LEARNING RESULTS ---")
    print(f"Accuracy: {accuracy_score(y_true_deep, y_pred_deep):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true_deep, y_pred_deep))
    
    # Save only the raw metrics to a file
    with open('evaluation_report.txt', 'w') as f:
        f.write("=== EVALUATION METRICS ===\n\n")
        
        f.write("1. TRADITIONAL APPROACH\n")
        f.write(f"Accuracy: {accuracy_score(y_true_trad, y_pred_trad):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true_trad, y_pred_trad, target_names=classes))
        
        f.write("\n\n2. DEEP LEARNING APPROACH\n")
        f.write(f"Accuracy: {accuracy_score(y_true_deep, y_pred_deep):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true_deep, y_pred_deep, target_names=classes))
        
    print("\nMetrics saved to evaluation_report.txt")