import torch
from torchvision import datasets, transforms
import os
import torch.nn as nn
from torchvision import models
import copy
import torch.optim as optim


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    print(f'Best val Acc: {best_acc:4f}')


    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'data_split'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    # Load pre-trained MobileNetV2 
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze initial layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classifier 
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
   
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters of the classifier
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)
    
    print("Starting Deep Learning training (Transfer Learning)...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/deep_mobilenet.pth')
    print("Model saved to models/deep_mobilenet.pth")