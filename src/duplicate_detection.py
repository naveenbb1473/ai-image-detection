import os
import json
import time
import hashlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import imagehash
from tqdm import tqdm

DATASET_DIRS = {
    "real": "dataset/real",
    "ai_generated": "dataset/ai_generated"
}

def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except Exception:
        return None

def compute_sha256(file_path):
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def compute_dhash(file_path):
    try:
        with Image.open(file_path) as img:
            h = imagehash.dhash(img)
            # Convert ImageHash object to 64-bit integer
            return int(str(h), 16)
    except Exception:
        return None

def get_lsh_chunks(hash_int):
    # Split 64-bit hash into 4 chunks of 16-bits
    mask = 0xFFFF
    c1 = (hash_int >> 48) & mask
    c2 = (hash_int >> 32) & mask
    c3 = (hash_int >> 16) & mask
    c4 = hash_int & mask
    return (c1, c2, c3, c4)

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def run_duplicate_detection():
    print("=" * 50)
    print("STARTING DATA QUALITY AND DUPLICATE DETECTION")
    print("=" * 50)
    
    # 1. Collect all valid image paths from disk
    print("Collecting files...")
    all_files = []
    for label, dir_path in DATASET_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue
        for entry in os.scandir(dir_path):
            if entry.is_file() and not entry.name.endswith(".json"):
                all_files.append((entry.path, label))
                
    total_files = len(all_files)
    print(f"Found {total_files} images to process.")
    
    # 2. Exact duplicate detection using file size groups first to minimize I/O
    print("Grouping files by size for exact duplicate detection...")
    size_groups = defaultdict(list)
    for path, label in all_files:
        sz = get_file_size(path)
        if sz is not None:
            size_groups[sz].append(path)
            
    candidate_paths = []
    for sz, paths in size_groups.items():
        if len(paths) > 1:
            candidate_paths.extend(paths)
            
    print(f"Found {len(candidate_paths)} files sharing sizes with other files. Computing SHA-256...")
    
    sha256_map = {}
    num_workers = min(16, (os.cpu_count() or 4) * 2)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_sha256, path): path for path in candidate_paths}
        for future in tqdm(as_completed(futures), total=len(candidate_paths), desc="Hashing files"):
            path = futures[future]
            h = future.result()
            if h is not None:
                sha256_map[path] = h
                
    # Group by SHA-256 hash
    hash_groups = defaultdict(list)
    for path, h in sha256_map.items():
        hash_groups[h].append(path)
        
    exact_duplicates = {}
    total_duplicate_images = 0
    for h, paths in hash_groups.items():
        if len(paths) > 1:
            exact_duplicates[h] = paths
            total_duplicate_images += len(paths)
            
    print(f"Found {len(exact_duplicates)} groups of exact duplicate files (total {total_duplicate_images} images).")
    
    # 3. Near-duplicate detection using dHash
    print("Computing perceptual dHash for all images...")
    dhash_map = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_dhash, path): path for path, label in all_files}
        for future in tqdm(as_completed(futures), total=total_files, desc="Perceptual hashing"):
            path = futures[future]
            dh = future.result()
            if dh is not None:
                dhash_map[path] = dh
                
    print(f"Successfully computed dHash for {len(dhash_map)} images.")
    
    # 4. LSH-based Near-duplicate clustering (Hamming distance <= 3)
    print("Indexing hashes with Locality Sensitive Hashing (LSH)...")
    lsh_buckets = [defaultdict(list) for _ in range(4)]
    
    for path, dh in dhash_map.items():
        chunks = get_lsh_chunks(dh)
        for i in range(4):
            lsh_buckets[i][chunks[i]].append(path)
            
    print("Querying LSH buckets for near-duplicates (Hamming distance <= 3)...")
    near_duplicates = []
    seen_pairs = set()
    
    # To prevent O(M^2) explosions, cap the bucket size for comparisons
    MAX_BUCKET_SIZE = 200
    
    # Iterate through all images to find candidates
    for path, dh in tqdm(dhash_map.items(), desc="Finding matches"):
        chunks = get_lsh_chunks(dh)
        candidates = set()
        for i in range(4):
            bucket = lsh_buckets[i][chunks[i]]
            if len(bucket) <= MAX_BUCKET_SIZE:
                candidates.update(bucket)
                
        for cand in candidates:
            if cand == path:
                continue
            pair = tuple(sorted([path, cand]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            # Compute Hamming distance
            dist = hamming_distance(dh, dhash_map[cand])
            if dist <= 3:
                # Store match
                near_duplicates.append({
                    "image_1": pair[0],
                    "image_2": pair[1],
                    "distance": dist
                })
                
    print(f"Found {len(near_duplicates)} potential near-duplicate pairs (Hamming distance <= 3).")
    
    # Group near duplicates into clusters/groups for easy viewing
    # Using a simple Union-Find / Disjoint Set structure to group them
    parent = {}
    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]
        
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
            
    for pair in near_duplicates:
        union(pair["image_1"], pair["image_2"])
        
    near_duplicate_groups = defaultdict(list)
    for node in parent:
        root = find(node)
        near_duplicate_groups[root].append(node)
        
    # Filter groups to only those with size > 1
    near_duplicate_clusters = [paths for paths in near_duplicate_groups.values() if len(paths) > 1]
    total_near_duplicate_images = sum(len(c) for c in near_duplicate_clusters)
    
    print(f"Grouped near-duplicates into {len(near_duplicate_clusters)} clusters (total {total_near_duplicate_images} images).")
    
    # 5. Compile and save reports
    os.makedirs("results", exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_scanned": total_files,
        "exact_duplicates_groups_count": len(exact_duplicates),
        "exact_duplicates_total_images": total_duplicate_images,
        "near_duplicates_groups_count": len(near_duplicate_clusters),
        "near_duplicates_total_images": total_near_duplicate_images,
        "exact_duplicate_groups_details": list(exact_duplicates.values())[:100],  # sample first 100
        "near_duplicate_clusters_details": near_duplicate_clusters[:100]           # sample first 100
    }
    
    with open("results/duplicate_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    text_report = f"""==================================================
DATA QUALITY & DUPLICATE DETECTION REPORT
==================================================
Timestamp: {report['timestamp']}
Total Images Scanned: {report['total_images_scanned']}

EXACT DUPLICATES (File Hash SHA-256):
- Duplicate groups found: {report['exact_duplicates_groups_count']}
- Total duplicate images involved: {report['exact_duplicates_total_images']}

NEAR-DUPLICATES (Perceptual dHash, Hamming Dist <= 3):
- Potential duplicate clusters found: {report['near_duplicates_groups_count']}
- Total near-duplicate images involved: {report['near_duplicates_total_images']}

DETECTION DETAILS:
- Exact duplicate groups (first 10 samples listed):
"""
    for idx, group in enumerate(list(exact_duplicates.values())[:10]):
        text_report += f"  Group {idx+1}: {', '.join(group)}\n"
        
    text_report += "\n- Near-duplicate clusters (first 10 samples listed):\n"
    for idx, cluster in enumerate(near_duplicate_clusters[:10]):
        text_report += f"  Cluster {idx+1}: {', '.join(cluster)}\n"
        
    text_report += "\n==================================================\n"
    
    with open("results/duplicate_report.txt", "w") as f:
        f.write(text_report)
        
    print("Duplicate report saved successfully:")
    print(" - results/duplicate_report.json")
    print(" - results/duplicate_report.txt")
    print("\nSummary:")
    print(text_report)

if __name__ == "__main__":
    run_duplicate_detection()
