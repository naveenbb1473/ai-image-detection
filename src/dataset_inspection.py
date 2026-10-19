import os
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

DATASET_DIRS = {
    "real": "dataset/real",
    "ai_generated": "dataset/ai_generated"
}

def inspect_single_image(args):
    file_path, label = args
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        file_size = os.path.getsize(file_path)
    except Exception:
        return {"status": "missing", "path": file_path, "label": label}
        
    # Mark files that are extremely small as corrupted (e.g. empty files)
    if file_size < 100:
        return {
            "status": "corrupted",
            "reason": "extremely_small",
            "path": file_path,
            "label": label,
            "size": file_size,
            "ext": ext
        }
        
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format
            img_mode = img.mode
            
            # verify reads the file header/structure to check integrity
            img.verify()
            
            return {
                "status": "valid",
                "path": file_path,
                "label": label,
                "ext": ext,
                "format": img_format,
                "width": width,
                "height": height,
                "mode": img_mode,
                "size": file_size
            }
    except Exception as e:
        return {
            "status": "corrupted",
            "reason": f"exception: {str(e)}",
            "path": file_path,
            "label": label,
            "size": file_size,
            "ext": ext
        }

def run_inspection():
    print("=" * 50)
    print("STARTING DATASET INSPECTION")
    print("=" * 50)
    
    # 1. Scan directory contents
    print("Scanning directories for files...")
    all_files = []
    for label, dir_path in DATASET_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue
        # Only collect actual image files, ignore JSON metadata files
        for entry in os.scandir(dir_path):
            if entry.is_file() and not entry.name.endswith(".json"):
                all_files.append((entry.path, label))
                
    total_files = len(all_files)
    print(f"Found {total_files} total image files on disk.")
    
    # 2. Load JSON metadata files for AI-generated images
    print("Loading AI-generated metadata JSON files...")
    metadata_map = {}
    ai_dir = DATASET_DIRS["ai_generated"]
    if os.path.exists(ai_dir):
        json_files = [f for f in os.listdir(ai_dir) if f.endswith(".json")]
        print(f"Found {len(json_files)} metadata JSON files.")
        for jf in json_files:
            try:
                with open(os.path.join(ai_dir, jf), "r") as f:
                    data = json.load(f)
                    metadata_map.update(data)
            except Exception as e:
                print(f"Error loading {jf}: {e}")
    print(f"Loaded metadata for {len(metadata_map)} AI-generated images.")
    
    # 3. Perform multithreaded inspection
    results = []
    num_workers = min(16, (os.cpu_count() or 4) * 2)
    print(f"Inspecting images using {num_workers} threads...")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(inspect_single_image, item): item for item in all_files}
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing"):
            results.append(future.result())
            
    elapsed_time = time.time() - start_time
    print(f"Image inspection completed in {elapsed_time:.2f} seconds.")
    
    # 4. Process inspection results
    real_count = 0
    ai_count = 0
    valid_count = 0
    corrupted_count = 0
    missing_count = 0
    
    ext_counter = Counter()
    format_counter = Counter()
    res_counter = Counter()
    mode_counter = Counter()
    
    corrupted_files_list = []
    missing_files_list = []
    all_basenames = []
    
    for r in results:
        label = r["label"]
        path = r["path"]
        basename = os.path.basename(path)
        all_basenames.append(basename)
        
        if label == "real":
            real_count += 1
        elif label == "ai_generated":
            ai_count += 1
            
        status = r["status"]
        if status == "valid":
            valid_count += 1
            ext_counter[r["ext"]] += 1
            format_counter[r["format"]] += 1
            res_counter[f"{r['width']}x{r['height']}"] += 1
            mode_counter[r["mode"]] += 1
        elif status == "corrupted":
            corrupted_count += 1
            corrupted_files_list.append({"path": path, "reason": r["reason"], "size": r.get("size", 0)})
        elif status == "missing":
            missing_count += 1
            missing_files_list.append(path)
            
    # Check for duplicate basenames
    basename_counts = Counter(all_basenames)
    duplicate_groups = {name: count for name, count in basename_counts.items() if count > 1}
    
    # Check what metadata fields and parameters are present in JSON
    metadata_keys_found = set()
    samplers_found = Counter()
    steps_found = Counter()
    cfg_found = Counter()
    
    # Let's match the valid AI images with JSON keys
    matched_metadata_count = 0
    for r in results:
        if r["status"] == "valid" and r["label"] == "ai_generated":
            basename = os.path.basename(r["path"])
            if basename in metadata_map:
                matched_metadata_count += 1
                meta = metadata_map[basename]
                metadata_keys_found.update(meta.keys())
                if "sa" in meta:
                    samplers_found[meta["sa"]] += 1
                if "st" in meta:
                    steps_found[meta["st"]] += 1
                if "c" in meta:
                    cfg_found[meta["c"]] += 1

    # Determine Heuristic AI Generator Source
    # Since they have sampler (sa), steps (st), cfg (c), seed (se) and prompt (p),
    # this matches Stable Diffusion parameters. Let's document this.
    inferred_generator = "Stable Diffusion (via prompt/sampler metadata)" if len(metadata_map) > 0 else "Unknown (No Metadata)"
    
    # Compile Report
    os.makedirs("results", exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files_scanned": total_files,
        "real_images": real_count,
        "ai_generated_images": ai_count,
        "valid_images": valid_count,
        "corrupted_images": corrupted_count,
        "missing_files": missing_count,
        "image_extensions": dict(ext_counter),
        "image_formats": dict(format_counter),
        "image_resolutions_top20": dict(res_counter.most_common(20)),
        "color_modes": dict(mode_counter),
        "metadata": {
            "total_json_entries": len(metadata_map),
            "matched_to_images": matched_metadata_count,
            "unique_metadata_fields": list(metadata_keys_found),
            "samplers": dict(samplers_found),
            "steps": dict(steps_found),
            "cfg_scales": dict(cfg_found),
            "inferred_generator": inferred_generator
        },
        "duplicate_filenames_count": len(duplicate_groups),
        "duplicate_filenames_sample": list(duplicate_groups.keys())[:20],
        "corrupted_files_details": corrupted_files_list[:100]
    }
    
    # Save Report JSON
    with open("results/dataset_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Generate formatted text report
    text_report = f"""==================================================
DATASET REPORT
==================================================
Timestamp: {report['timestamp']}
Total Files Scanned: {report['total_files_scanned']}

CLASS DISTRIBUTION:
- Real images: {report['real_images']}
- AI-generated images: {report['ai_generated_images']}
- Class Balance Ratio: {report['real_images'] / max(1, report['ai_generated_images']):.4f}

INTEGRITY STATUS:
- Valid images: {report['valid_images']}
- Corrupted images: {report['corrupted_images']}
- Missing files: {report['missing_files']}

FILE EXTENSIONS CONFORMITY:
"""
    for ext, count in sorted(ext_counter.items(), key=lambda x: x[1], reverse=True):
        text_report += f"- {ext.upper() or 'No Ext'}: {count} ({count/max(1, valid_count)*100:.2f}%)\n"
        
    text_report += "\nIMAGE FORMATS (Pillow decoded):\n"
    for fmt, count in sorted(format_counter.items(), key=lambda x: x[1], reverse=True):
        text_report += f"- {fmt}: {count}\n"

    text_report += "\nCOLOR MODES:\n"
    for mode, count in sorted(mode_counter.items(), key=lambda x: x[1], reverse=True):
        text_report += f"- {mode}: {count}\n"
        
    text_report += "\nTOP 10 RESOLUTIONS:\n"
    for res, count in res_counter.most_common(10):
        text_report += f"- {res}: {count} ({count/max(1, valid_count)*100:.2f}%)\n"
        
    text_report += f"\nDUPLICATE FILENAMES COUNT: {report['duplicate_filenames_count']}\n"
    if report['duplicate_filenames_count'] > 0:
        text_report += f"Sample duplicate filenames (first 20): {', '.join(report['duplicate_filenames_sample'])}\n"
        
    text_report += "\nAI GENERATOR / METADATA ANALYSIS:\n"
    text_report += f"- Total JSON Metadata entries: {report['metadata']['total_json_entries']}\n"
    text_report += f"- Matched to images on disk: {report['metadata']['matched_to_images']}\n"
    text_report += f"- Inferred generator source: {report['metadata']['inferred_generator']}\n"
    text_report += f"- Metadata fields: {', '.join(report['metadata']['unique_metadata_fields'])}\n"
    text_report += f"- Top Samplers: {sorted(report['metadata']['samplers'].items(), key=lambda x: x[1], reverse=True)[:5]}\n"
    text_report += f"- Top Steps: {sorted(report['metadata']['steps'].items(), key=lambda x: x[1], reverse=True)[:5]}\n"
    
    if corrupted_count > 0:
        text_report += f"\nWARNING: Found {corrupted_count} corrupted files. First 10 listed below:\n"
        for detail in corrupted_files_list[:10]:
            text_report += f"- Path: {detail['path']} | Reason: {detail['reason']} | Size: {detail['size']} bytes\n"
            
    text_report += "\n==================================================\n"
    
    with open("results/dataset_report.txt", "w") as f:
        f.write(text_report)
        
    print("Dataset report saved successfully:")
    print(" - results/dataset_report.json")
    print(" - results/dataset_report.txt")
    print("\nSummary:")
    print(text_report)

if __name__ == "__main__":
    run_inspection()
