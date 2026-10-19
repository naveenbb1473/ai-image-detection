import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from model import LinearClassifier, MLPClassifier

def evaluate_on_test_set(model_path, split_name="test"):
    print("=" * 50)
    print(f"EVALUATING MODEL: {model_path} ON '{split_name.upper()}' SET")
    print("=" * 50)
    
    # 1. Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
        
    checkpoint = torch.load(model_path)
    model_type = checkpoint.get("model_type", "linear")
    embed_dim = checkpoint.get("embed_dim", 384)
    epoch = checkpoint.get("epoch", "unknown")
    
    print(f"Loaded checkpoint trained for {epoch} epochs.")
    print(f"Model Type: {model_type.upper()} | Embedding Dimension: {embed_dim}")
    
    # 2. Reconstruct Model Architecture
    if model_type == "linear":
        model = LinearClassifier(embed_dim=embed_dim, num_classes=2)
    elif model_type == "mlp":
        model = MLPClassifier(embed_dim=embed_dim, num_classes=2)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Load Test Features and Labels
    test_features_path = f"features/{split_name}_features.pt"
    test_labels_path = f"features/{split_name}_labels.pt"
    
    if not os.path.exists(test_features_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"Feature/label files for split '{split_name}' not found. Run extraction first.")
        
    test_features = torch.load(test_features_path)
    test_labels = torch.load(test_labels_path)
    print(f"Loaded test set with {test_features.shape[0]} samples.")
    
    # 4. Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_features = test_features.to(device)
    
    with torch.no_grad():
        outputs = model(test_features)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    y_true = test_labels.numpy()
    
    # 5. Compute Metrics
    accuracy = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
    roc_auc = roc_auc_score(y_true, probs[:, 1])
    
    print("-" * 50)
    print("CLASSIFICATION METRICS:")
    print(f"- Accuracy:  {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall:    {recall:.4f}")
    print(f"- F1-Score:  {f1:.4f}")
    print(f"- ROC-AUC:   {roc_auc:.4f}")
    print("-" * 50)
    
    metrics = {
        "model_type": model_type,
        "epoch": epoch,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    
    # 6. Save text classification report
    os.makedirs("results", exist_ok=True)
    report_text = classification_report(y_true, preds, target_names=["REAL", "AI_GENERATED"])
    print("Classification Report:")
    print(report_text)
    
    # Save files with specific names for comparisons, and as baseline default files
    report_suffix = f"_{model_type}" if "best_model.pth" not in model_path else ""
    
    with open(f"results/classification_report{report_suffix}.txt", "w") as f:
        f.write(report_text)
        
    with open(f"results/metrics{report_suffix}.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 7. Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["REAL", "AI_GENERATED"], 
        yticklabels=["REAL", "AI_GENERATED"]
    )
    plt.title(f"Confusion Matrix - {model_type.upper()} Classifier")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = f"results/confusion_matrix{report_suffix}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # 8. Generate and save ROC Curve plot
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_type.upper()}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = f"results/roc_curve{report_suffix}.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classification head on untouched test set")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint model file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (test or validation)")
    args = parser.parse_args()
    
    evaluate_on_test_set(args.model_path, split_name=args.split)

if __name__ == "__main__":
    main()
