import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AIDetectionDataset(Dataset):
    """
    Dataset class for AI-Generated Image Detection.
    Loads paths and labels from split CSV files and returns preprocessed image tensors and labels.
    """
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = int(self.df.iloc[idx]["label"])
        
        try:
            # Open image and convert to RGB format
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback placeholder image in case of runtime I/O issues
            print(f"Error loading image {img_path}: {e}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dinov2_transforms(img_size=224):
    """
    Returns standard preprocessing transforms for DINOv2.
    Resizes images to a square multiple of 14 (standard is 224) and normalizes using ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def test_dataset_loading():
    """
    Basic verification test to check dataset loading and shapes.
    """
    print("Testing dataset implementation...")
    train_csv = "splits/train.csv"
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} does not exist. Please run splitting first.")
        return
        
    transform = get_dinov2_transforms(img_size=224)
    dataset = AIDetectionDataset(csv_file=train_csv, transform=transform)
    
    print(f"Successfully loaded dataset with {len(dataset)} samples.")
    
    # Load first sample
    img_tensor, label = dataset[0]
    print(f"Sample 0 - Image shape: {img_tensor.shape} | Label: {label}")
    print(f"Tensor type: {img_tensor.dtype} | Min value: {img_tensor.min():.4f} | Max value: {img_tensor.max():.4f}")
    
    # Verify shape
    assert img_tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {img_tensor.shape}"
    print("Dataset verification test: PASSED")

if __name__ == "__main__":
    test_dataset_loading()
