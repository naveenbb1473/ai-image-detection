import torch
import torch.nn as nn

class DINOv2Classifier(nn.Module):
    """
    End-to-end model combining frozen DINOv2 backbone with a classification head.
    Mainly used for validation, testing, final prediction pipeline, and XAI.
    """
    def __init__(self, backbone_name='dinov2_vits14', classifier_type='linear', num_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        
        # Load DINOv2 from PyTorch Hub
        print(f"Loading DINOv2 backbone ({backbone_name}) from PyTorch Hub...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        
        if freeze_backbone:
            print("Freezing DINOv2 backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Determine embedding dimension dynamically
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.embed_dim = dummy_output.shape[1]
            
        print(f"Inferred backbone embedding dimension: {self.embed_dim}")
        
        self.classifier_type = classifier_type
        if classifier_type == 'linear':
            self.classifier = nn.Linear(self.embed_dim, num_classes)
        elif classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def forward(self, x):
        # Extract features (use no_grad block for frozen backbone)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
            
        # Classify features
        return self.classifier(features)

class LinearClassifier(nn.Module):
    """
    Simple Linear Classification head for training directly on pre-extracted features.
    """
    def __init__(self, embed_dim=384, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class MLPClassifier(nn.Module):
    """
    Lightweight Multi-Layer Perceptron classification head for training on pre-extracted features.
    """
    def __init__(self, embed_dim=384, num_classes=2, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)

def test_model_loading():
    """
    Test script to verify loading, device mapping, and forward pass sizes.
    """
    print("Testing DINOv2 model loading and forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    # Load end-to-end model
    model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type='linear', num_classes=2)
    model = model.to(device)
    model.eval()
    
    # Generate random test tensor (batch_size=2, RGB channels=3, size=224x224)
    x = torch.randn(2, 3, 224, 224, device=device)
    
    with torch.no_grad():
        out = model(x)
        
    print(f"Forward pass output shape: {out.shape}")
    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"
    print("Model verification test: PASSED")

if __name__ == "__main__":
    test_model_loading()
