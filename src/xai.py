import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from dataset import get_dinov2_transforms
from model import DINOv2Classifier, LinearClassifier, MLPClassifier

# Global variable to capture input to the attention block via hook
attn_input = None

def attention_hook_fn(module, input_args, output):
    global attn_input
    attn_input = input_args[0]

def compute_attention_map(model, img_tensor, device):
    """
    Attaches a forward hook to the last block's attention layer, runs a forward pass,
    and manually computes the self-attention weights from the [CLS] token to all patches.
    """
    global attn_input
    attn_input = None
    
    # 1. Register forward hook on the attention layer of the last block
    attn_module = model.backbone.blocks[-1].attn
    handle = attn_module.register_forward_hook(attention_hook_fn)
    
    # 2. Run model forward pass
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_label].item()
        
    # 3. Remove hook
    handle.remove()
    
    if attn_input is None:
        raise RuntimeError("Failed to capture attention layer input via hook.")
        
    # 4. Manually compute the attention matrix from the captured input
    # Shape of attn_input: [B, N, C] where B=1, N=num_patches+1, C=embed_dim
    B, N, C = attn_input.shape
    num_heads = attn_module.num_heads
    head_dim = C // num_heads
    scale = head_dim ** -0.5
    
    # Project input to Q, K, V
    qkv = attn_module.qkv(attn_input).reshape(B, N, 3, num_heads, head_dim)
    q, k, v = torch.unbind(qkv, 2)
    
    # Transpose to [B, num_heads, N, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    
    # Compute self-attention weights: [B, num_heads, N, N]
    attn_weights = (q @ k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # Extract attention weights from [CLS] token (index 0) to all spatial patches (index 1 to N-1)
    # Shape: [num_heads, num_patches]
    cls_attention = attn_weights[0, :, 0, 1:]
    
    # Average attention maps across all heads
    # Shape: [num_patches]
    avg_attention = cls_attention.mean(dim=0)
    
    # Reshape to 2D grid (for 224x224 input and patch size 14, grid size is 224/14 = 16)
    grid_size = int(avg_attention.shape[-1] ** 0.5)
    attn_map_2d = avg_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    return attn_map_2d, pred_label, confidence

def generate_heatmap_overlay(orig_img, attn_map_2d):
    """
    Interpolates the attention map back to the original image size and overlays it as a heatmap.
    """
    # Resize attention map using bilinear interpolation
    attn_pil = Image.fromarray((attn_map_2d / (attn_map_2d.max() + 1e-8) * 255).astype(np.uint8))
    attn_resized = attn_pil.resize(orig_img.size, Image.Resampling.BILINEAR)
    attn_array = np.array(attn_resized)
    
    # Create colormap
    colormap = plt.get_cmap('jet')
    heatmap = colormap(attn_array / 255.0)[:, :, :3]  # drop alpha channel
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    
    # Blend original image and heatmap (alpha blend)
    overlay = Image.blend(orig_img, heatmap, alpha=0.55)
    return overlay, attn_resized

def run_faithfulness_perturbation(image_input, model, device, threshold_pct=0.15, attn_map=None):
    """
    Performs perturbation-based faithfulness analysis on an input image (PIL Image or file path).
    Generates:
      - 3 visual comparison images:
          1. Before Perturbation (Original 224x224)
          2. After Attribution Masking (Top % attended patches masked)
          3. After Random Masking (Random % patches masked)
      - Quantitative metrics:
          - Baseline, Attribution, and Random logits & confidences
          - Logit drop and confidence drop comparisons
          - Faithfulness test verdict
    """
    if isinstance(image_input, str):
        orig_img = Image.open(image_input).convert("RGB")
        image_path = image_input
    else:
        orig_img = image_input.convert("RGB")
        image_path = None

    transform = get_dinov2_transforms(img_size=224)
    img_tensor = transform(orig_img).unsqueeze(0).to(device)

    # 1. Baseline prediction
    model.eval()
    with torch.no_grad():
        base_out = model(img_tensor)
        base_probs = torch.softmax(base_out, dim=1)
        pred_class = torch.argmax(base_out, dim=1).item()
        base_conf = base_probs[0, pred_class].item()
        base_logit = base_out[0, pred_class].item()

    pred_label_str = "AI_GENERATED" if pred_class == 1 else "REAL"

    # 2. Compute attention map if not passed
    if attn_map is None:
        attn_map, _, _ = compute_attention_map(model, img_tensor, device)

    # 3. Identify patch indices
    flat_attn = attn_map.flatten()
    num_patches = len(flat_attn)
    num_to_mask = max(1, int(num_patches * threshold_pct))

    # Top attended patches
    top_indices = np.argsort(flat_attn)[::-1][:num_to_mask]

    # Random patches (excluding top attended ones)
    remaining_indices = list(set(range(num_patches)) - set(top_indices))
    if len(remaining_indices) < num_to_mask:
        remaining_indices = list(range(num_patches))
    random_indices = np.random.choice(remaining_indices, size=num_to_mask, replace=False)

    # Patch geometry (16x16 grid for 224x224 input and 14x14 patch size)
    patch_size = 14
    grid_size = 224 // patch_size

    def mask_tensor_patches(tensor, indices):
        masked = tensor.clone()
        for idx in indices:
            row = idx // grid_size
            col = idx % grid_size
            masked[0, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0
        return masked

    attn_masked_tensor = mask_tensor_patches(img_tensor, top_indices)
    rand_masked_tensor = mask_tensor_patches(img_tensor, random_indices)

    # Run predictions on perturbed tensors
    with torch.no_grad():
        attn_out = model(attn_masked_tensor)
        attn_probs = torch.softmax(attn_out, dim=1)
        attn_conf = attn_probs[0, pred_class].item()
        attn_logit = attn_out[0, pred_class].item()

        rand_out = model(rand_masked_tensor)
        rand_probs = torch.softmax(rand_out, dim=1)
        rand_conf = rand_probs[0, pred_class].item()
        rand_logit = rand_out[0, pred_class].item()

    # Calculate drops
    attn_conf_drop = base_conf - attn_conf
    rand_conf_drop = base_conf - rand_conf
    attn_logit_drop = base_logit - attn_logit
    rand_logit_drop = base_logit - rand_logit

    passed_conf_test = bool(attn_conf_drop > rand_conf_drop)
    passed_logit_test = bool(attn_logit_drop > rand_logit_drop)

    # Generate PIL visualizations (224x224)
    orig_224 = orig_img.resize((224, 224), Image.Resampling.BILINEAR)

    def create_masked_pil(base_pil, indices, mask_color=(20, 20, 20)):
        img_np = np.array(base_pil).copy()
        for idx in indices:
            r = idx // grid_size
            c = idx % grid_size
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            img_np[y0:y1, x0:x1] = mask_color
        return Image.fromarray(img_np)

    attn_masked_pil = create_masked_pil(orig_224, top_indices)
    rand_masked_pil = create_masked_pil(orig_224, random_indices)

    return {
        "image_path": image_path,
        "orig_img_224": orig_224,
        "attn_masked_img": attn_masked_pil,
        "rand_masked_img": rand_masked_pil,
        "pred_class": pred_class,
        "pred_label": pred_label_str,
        "base_confidence": base_conf,
        "base_logit": base_logit,
        "attribution_masked_confidence": attn_conf,
        "attribution_masked_logit": attn_logit,
        "attribution_conf_drop": attn_conf_drop,
        "attribution_logit_drop": attn_logit_drop,
        "random_masked_confidence": rand_conf,
        "random_masked_logit": rand_logit,
        "random_conf_drop": rand_conf_drop,
        "random_logit_drop": rand_logit_drop,
        "passed_logit_test": passed_logit_test,
        "passed_conf_test": passed_conf_test,
        "threshold_pct": threshold_pct,
        "num_patches_masked": num_to_mask,
        "total_patches": num_patches,
        "top_indices": top_indices,
        "random_indices": random_indices
    }

def run_faithfulness_test(image_path, model, device, threshold_pct=0.15):
    """
    Performs a perturbation-based faithfulness check of the XAI explanation.
    Compares the model's confidence drop and logit drop when masking:
    1. Highly attended patches (Model-attribution based)
    2. Randomly selected patches of the same area
    """
    print(f"\nRunning XAI Faithfulness Check on: {os.path.basename(image_path)}")
    res = run_faithfulness_perturbation(image_path, model, device, threshold_pct=threshold_pct)
    
    print(f"Baseline Pred: {res['pred_label']} | Conf: {res['base_confidence']:.6f} | Logit: {res['base_logit']:.4f}")
    print("FAITHFULNESS ANALYSIS:")
    print(f"- Confidence with Attribution Masked: {res['attribution_masked_confidence']:.6f} (Drop: {res['attribution_conf_drop']:.6f})")
    print(f"- Confidence with Random Masked:      {res['random_masked_confidence']:.6f} (Drop: {res['random_conf_drop']:.6f})")
    print(f"- Logit with Attribution Masked:      {res['attribution_masked_logit']:.4f} (Drop: {res['attribution_logit_drop']:.4f})")
    print(f"- Logit with Random Masked:           {res['random_masked_logit']:.4f} (Drop: {res['random_logit_drop']:.4f})")
    print(f"- Faithfulness Test (Probability):     {'PASSED' if res['passed_conf_test'] else 'FAILED'} (Score: {res['attribution_conf_drop'] - res['random_conf_drop']:.6f})")
    print(f"- Faithfulness Test (Logit):           {'PASSED' if res['passed_logit_test'] else 'FAILED'} (Score: {res['attribution_logit_drop'] - res['random_logit_drop']:.4f})")
    
    return res

def save_xai_visualization(image_path, model, device, output_dir="results/xai"):
    """
    Generates and saves the composite XAI visualization (Original | Heatmap Overlay | Attention Mask).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess
    transform = get_dinov2_transforms(img_size=224)
    orig_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(orig_img).unsqueeze(0)
    
    # Compute attention map
    attn_map, pred_class, confidence = compute_attention_map(model, img_tensor, device)
    pred_str = "AI_GENERATED" if pred_class == 1 else "REAL"
    
    # Generate overlay
    overlay, attn_resized = generate_heatmap_overlay(orig_img, attn_map)
    
    # Plot composite visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 2. Raw Attention Heatmap
    im = axes[1].imshow(attn_map, cmap='jet')
    axes[1].set_title("DINOv2 Self-Attention Map (16x16)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Heatmap Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {pred_str}\nConfidence: {confidence*100:.2f}%")
    axes[2].axis("off")
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_xai_explanation.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"XAI visualization saved successfully to {output_path}")
    return output_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best MLP checkpoint model
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please train the classifier first.")
        return
        
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model_type = checkpoint.get("model_type", "mlp")
    
    # Instantiate the end-to-end DINOv2Classifier
    model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type=model_type, num_classes=2)
    
    # Map the trained classifier head weights from the checkpoint
    head_state_dict = checkpoint["model_state_dict"]
    mapped_state_dict = {}
    for k, v in head_state_dict.items():
        if model_type == "linear":
            new_k = k.replace("fc.", "classifier.")
        else:
            new_k = k.replace("mlp.", "classifier.")
        mapped_state_dict[new_k] = v
        
    model.load_state_dict(mapped_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Pick a sample image to visualize from the splits
    train_csv = "splits/train.csv"
    if not os.path.exists(train_csv):
        print("Error: train.csv not found.")
        return
        
    df = pd.read_csv(train_csv)
    # Pick the first AI image for visualization
    ai_samples = df[df["label"] == 1]
    if len(ai_samples) > 0:
        sample_path = ai_samples.iloc[0]["image_path"]
        
        # Save XAI visual explanation
        save_xai_visualization(sample_path, model, device)
        
        # Run faithfulness check
        run_faithfulness_test(sample_path, model, device, threshold_pct=0.15)
    else:
        print("No AI images found in train.csv.")

if __name__ == "__main__":
    # Import pandas only inside main test execution block to save importing time
    import pandas as pd
    main()
