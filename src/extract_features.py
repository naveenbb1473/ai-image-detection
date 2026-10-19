import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AIDetectionDataset, get_dinov2_transforms
from model import DINOv2Classifier

def extract_features_for_split(split_name, csv_path, model, transform, device, batch_size, limit=None, num_workers=0):
    """
    Extracts features for a single dataset split and returns them as tensors along with metadata.
    """
    print("-" * 50)
    print(f"Extracting features for split: {split_name} (Limit: {limit})")
    print("-" * 50)
    
    # Initialize dataset
    dataset = AIDetectionDataset(csv_file=csv_path, transform=transform)
    
    # If limit is set, truncate dataset
    if limit is not None and limit < len(dataset):
        dataset.df = dataset.df.iloc[:limit].reset_index(drop=True)
        print(f"Truncated split to first {limit} samples.")
        
    # Use DataLoader (configurable num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == 'cuda'))
    
    all_features = []
    all_labels = []
    all_paths = list(dataset.df["image_path"])
    
    model.eval()
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Processing {split_name}"):
            batch_imgs = batch_imgs.to(device)
            # DINOv2 forward pass to extract features directly from the backbone
            features = model.backbone(batch_imgs)
            
            # Move to CPU to save GPU VRAM
            all_features.append(features.cpu())
            all_labels.append(batch_labels)
            
    # Concatenate features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    return all_features, all_labels, all_paths

def main():
    parser = argparse.ArgumentParser(description="Extract features using pretrained DINOv2")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed per split (for testing)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Load model
    model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type='linear', num_classes=2)
    model = model.to(args.device)
    
    # 2. Preprocessing transform
    transform = get_dinov2_transforms(img_size=224)
    
    # Define splits to process
    splits = {
        "train": "splits/train.csv",
        "validation": "splits/validation.csv",
        "test": "splits/test.csv"
    }
    
    os.makedirs("features", exist_ok=True)
    
    # 3. Extract features for each split
    extraction_meta = {
        "backbone": model.backbone_name,
        "feature_dim": model.embed_dim,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": args.device,
        "splits": {}
    }
    
    for split_name, csv_path in splits.items():
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} does not exist. Skipping split {split_name}.")
            continue
            
        features, labels, paths = extract_features_for_split(
            split_name=split_name,
            csv_path=csv_path,
            model=model,
            transform=transform,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            num_workers=args.num_workers
        )
        
        # Save features and labels
        torch.save(features, f"features/{split_name}_features.pt")
        torch.save(labels, f"features/{split_name}_labels.pt")
        
        # Save paths to map index -> path
        with open(f"features/{split_name}_paths.json", "w") as f:
            json.dump(paths, f, indent=4)
            
        extraction_meta["splits"][split_name] = {
            "num_samples": len(paths),
            "features_file": f"features/{split_name}_features.pt",
            "labels_file": f"features/{split_name}_labels.pt",
            "paths_file": f"features/{split_name}_paths.json"
        }
        
    # Save extraction metadata
    # If it is a test run (with limit), save as metadata_test.json to avoid overwriting full runs
    meta_filename = "features/metadata_test.json" if args.limit is not None else "features/metadata.json"
    with open(meta_filename, "w") as f:
        json.dump(extraction_meta, f, indent=4)
        
    print(f"\nExtraction completed. Metadata saved to {meta_filename}")

if __name__ == "__main__":
    main()
