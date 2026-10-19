import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from model import LinearClassifier, MLPClassifier

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_metrics(y_true, y_pred, y_prob):
    """
    Computes classification metrics using scikit-learn.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    try:
        # ROC-AUC requires probability of positive class (label 1)
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        roc_auc = 0.5
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

def train_classifier(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Training on device: {device}")
    
    # 1. Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 2. Load pre-extracted features
    print("Loading pre-extracted DINOv2 features...")
    train_features = torch.load("features/train_features.pt")
    train_labels = torch.load("features/train_labels.pt")
    val_features = torch.load("features/validation_features.pt")
    val_labels = torch.load("features/validation_labels.pt")
    
    print(f"Loaded Train: {train_features.shape[0]} samples | Val: {val_features.shape[0]} samples")
    
    # 3. Create DataLoaders
    # Features are low-dimensional (384), so we can use larger batch sizes for faster training
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Initialize model
    embed_dim = train_features.shape[1]
    if args.model_type == "linear":
        model = LinearClassifier(embed_dim=embed_dim, num_classes=2)
    elif args.model_type == "mlp":
        model = MLPClassifier(embed_dim=embed_dim, num_classes=2, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
    model = model.to(device)
    print(f"Model architecture:\n{model}")
    
    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 6. Training Loop
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": []
    }
    
    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features_batch.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_targets = []
        val_probs = []
        
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)
                epoch_val_loss += loss.item() * features_batch.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        epoch_val_loss = epoch_val_loss / len(val_dataset)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        
        # Compute metrics
        metrics = evaluate_metrics(val_targets, val_preds, val_probs)
        
        # Log to history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])
        history["val_f1"].append(metrics["f1"])
        history["val_roc_auc"].append(metrics["roc_auc"])
        
        # Print metrics
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {metrics['accuracy']:.4f} | "
              f"Val F1: {metrics['f1']:.4f} | "
              f"Val AUC: {metrics['roc_auc']:.4f}")
              
        # Checkpointing based on validation F1 score
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_val_loss = epoch_val_loss
            # Construct checkpoint metadata
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_type": args.model_type,
                "embed_dim": embed_dim,
                "val_f1": best_val_f1,
                "val_accuracy": metrics["accuracy"],
                "val_roc_auc": metrics["roc_auc"]
            }
            # Save best model path
            checkpoint_path = f"checkpoints/best_{args.model_type}_model.pth"
            torch.save(checkpoint, checkpoint_path)
            # Duplicate as general best_model.pth for standard inference scripts
            torch.save(checkpoint, "checkpoints/best_model.pth")
            print(f"--> Saved best model checkpoint to {checkpoint_path} (F1: {best_val_f1:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping check
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement in {args.patience} epochs).")
            break
            
    # Save training history JSON
    history_path = f"results/training_history_{args.model_type}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining completed! Saved history to {history_path}")
    print(f"Best Validation F1: {best_val_f1:.4f} (associated Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification head on extracted DINOv2 features")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "mlp"], help="Classifier architecture")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MLP")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for MLP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()
    
    train_classifier(args)
