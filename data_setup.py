import os
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split, split_images in splits.items():
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, img))
        
        print(f"Class {cls}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

if __name__ == "__main__":
    random.seed(42)
    source = 'dataset'
    output = 'data_split'
    split_dataset(source, output)
    print("Dataset split complete.")
