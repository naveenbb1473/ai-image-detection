import os
import argparse
import torch
from PIL import Image

from dataset import get_dinov2_transforms
from model import DINOv2Classifier, LinearClassifier, MLPClassifier
from xai import compute_attention_map, save_xai_visualization

def predict_single_image(image_path, model_path, run_xai=False, device="cpu"):
    """
    Predicts whether a single image is REAL or AI_GENERATED using the best model checkpoint.
    Optionally saves its XAI visualization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    # 1. Load checkpoint and determine architecture
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "mlp")
    
    # 2. Instantiate and load end-to-end model
    model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type=model_type, num_classes=2, freeze_backbone=True)
    
    # Map the weights
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
    
    # 3. Load and preprocess image
    transform = get_dinov2_transforms(img_size=224)
    orig_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(orig_img).unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    real_prob = probs[0].item()
    ai_prob = probs[1].item()
    
    pred_label = 1 if ai_prob > real_prob else 0
    prediction = "AI_GENERATED" if pred_label == 1 else "REAL"
    confidence = ai_prob if pred_label == 1 else real_prob
    
    print("\n" + "=" * 40)
    print("AI IMAGE DETECTION PREDICTION RESULT")
    print("=" * 40)
    print(f"Image File:     {os.path.basename(image_path)}")
    print(f"Prediction:     {prediction}")
    print(f"Confidence:     {confidence*100:.2f}%")
    print("-" * 40)
    print(f"AI Probability:   {ai_prob:.6f}")
    print(f"Real Probability: {real_prob:.6f}")
    print("=" * 40 + "\n")
    
    result = {
        "image_path": image_path,
        "prediction": prediction,
        "confidence": confidence,
        "ai_probability": ai_prob,
        "real_probability": real_prob
    }
    
    # 5. Handle optional XAI visual generation
    if run_xai:
        print("Generating XAI attention map...")
        xai_output_path = save_xai_visualization(image_path, model, device)
        result["xai_visualization_path"] = xai_output_path
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Run prediction on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--xai", action="store_true", help="Generate and save XAI visual explanation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()
    
    predict_single_image(
        image_path=args.image_path,
        model_path=args.model_path,
        run_xai=args.xai,
        device=args.device
    )

if __name__ == "__main__":
    main()
