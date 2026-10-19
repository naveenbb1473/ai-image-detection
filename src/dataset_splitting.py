import os
import json
import random
import pandas as pd

DUPLICATE_REPORT_PATH = "results/duplicate_report.json"
DATASET_DIRS = {
    "real": "dataset/real",
    "ai_generated": "dataset/ai_generated"
}
SPLITS_DIR = "splits"

# Disjoint Set Union (DSU) to group overlapping duplicate/near-duplicate sets
class DSU:
    def __init__(self):
        self.parent = {}
        
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry

def run_dataset_splitting():
    print("=" * 50)
    print("STARTING DATASET SPLITTING WITH LEAKAGE PREVENTION")
    print("=" * 50)
    
    # 1. Load duplicate report
    if not os.path.exists(DUPLICATE_REPORT_PATH):
        print(f"Error: {DUPLICATE_REPORT_PATH} not found. Please run duplicate detection first.")
        return
        
    with open(DUPLICATE_REPORT_PATH, "r") as f:
        dup_report = json.load(f)
        
    # Get all duplicate groups and clusters
    exact_groups = dup_report.get("exact_duplicate_groups_details", [])
    near_clusters = dup_report.get("near_duplicate_clusters_details", [])
    
    # 2. Build disjoint sets of duplicates
    dsu = DSU()
    for group in exact_groups:
        for i in range(len(group) - 1):
            dsu.union(group[i], group[i+1])
            
    for cluster in near_clusters:
        for i in range(len(cluster) - 1):
            dsu.union(cluster[i], cluster[i+1])
            
    # Group duplicate paths by their root parent
    duplicate_clusters = {}
    for path in dsu.parent:
        root = dsu.find(path)
        if root not in duplicate_clusters:
            duplicate_clusters[root] = []
        duplicate_clusters[root].append(path)
        
    # Create a set of all duplicate image paths for quick lookup
    all_duplicate_paths = set(dsu.parent.keys())
    print(f"Collected {len(all_duplicate_paths)} duplicate/near-duplicate images grouped into {len(duplicate_clusters)} clusters.")
    
    # 3. Collect all images in the dataset
    print("Collecting files from disk...")
    real_images = []
    ai_images = []
    
    for label, dir_path in DATASET_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist.")
            return
        for entry in os.scandir(dir_path):
            if entry.is_file() and not entry.name.endswith(".json"):
                # Normalize paths to use forward slashes for consistency
                normalized_path = entry.path.replace("\\", "/")
                if label == "real":
                    real_images.append(normalized_path)
                else:
                    ai_images.append(normalized_path)
                    
    print(f"Total real images found: {len(real_images)}")
    print(f"Total AI images found: {len(ai_images)}")
    
    # 4. Separate images into independent and duplicate groups
    # Normalize duplicate paths to match normalized disk paths
    normalized_duplicate_paths = {p.replace("\\", "/") for p in all_duplicate_paths}
    
    real_independent = [p for p in real_images if p not in normalized_duplicate_paths]
    ai_independent = [p for p in ai_images if p not in normalized_duplicate_paths]
    
    real_duplicates = [p for p in real_images if p in normalized_duplicate_paths]
    ai_duplicates = [p for p in ai_images if p in normalized_duplicate_paths]
    
    print(f"Real independent: {len(real_independent)} | Real duplicate: {len(real_duplicates)}")
    print(f"AI independent: {len(ai_independent)} | AI duplicate: {len(ai_duplicates)}")
    
    # 5. Reproducible split of independent images
    # Fixed seed for reproducibility
    random.seed(42)
    random.shuffle(real_independent)
    random.shuffle(ai_independent)
    
    # 70% Train, 15% Val, 15% Test splits
    def split_list(lst):
        n = len(lst)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train = lst[:n_train]
        val = lst[n_train:n_train+n_val]
        test = lst[n_train+n_val:]
        return train, val, test
        
    real_train, real_val, real_test = split_list(real_independent)
    ai_train, ai_val, ai_test = split_list(ai_independent)
    
    # 6. Assign duplicate clusters to Train split to prevent leakage
    # By placing all duplicates in Train, we guarantee they do not leak into Val or Test.
    train_paths = real_train + ai_train + real_duplicates + ai_duplicates
    val_paths = real_val + ai_val
    test_paths = real_test + ai_test
    
    # Check data leakage explicitly
    set_train = set(train_paths)
    set_val = set(val_paths)
    set_test = set(test_paths)
    
    leak_train_val = set_train.intersection(set_val)
    leak_train_test = set_train.intersection(set_test)
    leak_val_test = set_val.intersection(set_test)
    
    print("-" * 50)
    print("LEAKAGE VERIFICATION:")
    print(f"- Train AND Validation overlap: {len(leak_train_val)}")
    print(f"- Train AND Test overlap: {len(leak_train_test)}")
    print(f"- Validation AND Test overlap: {len(leak_val_test)}")
    print("-" * 50)
    
    # Ensure there is absolutely no overlap
    assert len(leak_train_val) == 0, "Leakage detected between Train and Validation!"
    assert len(leak_train_test) == 0, "Leakage detected between Train and Test!"
    assert len(leak_val_test) == 0, "Leakage detected between Validation and Test!"
    print("VERIFICATION PASSED: No image path overlaps between splits.")
    
    # Build dataframes for splits
    # Add metadata fields: label, generator, source, category
    def create_split_df(paths):
        records = []
        for p in paths:
            # Label: 0 = REAL, 1 = AI_GENERATED
            label = 0 if "dataset/real" in p else 1
            
            # Metadata details
            if label == 0:
                generator = "Real_Camera"
                source = "Real"
                category = "Real"
            else:
                generator = "Stable_Diffusion"
                source = "Stable_Diffusion_Prompt_Dataset"
                category = "AI"
                
            records.append({
                "image_path": p,
                "label": label,
                "generator": generator,
                "source": source,
                "category": category
            })
        return pd.DataFrame(records)
        
    df_train = create_split_df(train_paths)
    df_val = create_split_df(val_paths)
    df_test = create_split_df(test_paths)
    
    # Maintain reproducible shuffling of final CSV rows
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(SPLITS_DIR, exist_ok=True)
    df_train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(SPLITS_DIR, "validation.csv"), index=False)
    df_test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    
    # Print Split Report
    total_samples = len(df_train) + len(df_val) + len(df_test)
    print("\n" + "=" * 50)
    print("DATASET SPLITS GENERATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Total split dataset size: {total_samples}")
    
    def print_split_stats(name, df):
        n = len(df)
        reals = len(df[df["label"] == 0])
        ais = len(df[df["label"] == 1])
        ratio = reals / max(1, ais)
        pct = (n / total_samples) * 100
        print(f"{name.upper()} split:")
        print(f"  Total: {n} ({pct:.2f}%)")
        print(f"  REAL: {reals} | AI_GENERATED: {ais} | Balance Ratio (R/AI): {ratio:.4f}")
        
    print_split_stats("train", df_train)
    print_split_stats("validation", df_val)
    print_split_stats("test", df_test)
    print("=" * 50)

if __name__ == "__main__":
    run_dataset_splitting()
