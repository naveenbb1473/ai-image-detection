import os
import sys
import unittest
import torch
from PIL import Image
import pandas as pd

# Ensure src is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import get_dinov2_transforms, AIDetectionDataset
from model import DINOv2Classifier, LinearClassifier, MLPClassifier
from xai import run_faithfulness_perturbation
from llm_explainer import generate_forensic_explanation

class TestAIDetectionPipeline(unittest.TestCase):
    
    def setUp(self):
        # 1. Create a dummy CSV manifest for dataset test
        self.dummy_csv = "dummy_manifest.csv"
        df = pd.DataFrame({
            "image_path": ["dummy_image1.png", "dummy_image2.png"],
            "label": [0, 1]
        })
        df.to_csv(self.dummy_csv, index=False)
        
        # 2. Create dummy images to avoid FileNotFoundError in dataset reading
        self.dummy_images = ["dummy_image1.png", "dummy_image2.png"]
        for img_path in self.dummy_images:
            img = Image.new("RGB", (256, 256), color=(255, 0, 0))
            img.save(img_path)
            
    def tearDown(self):
        # Clean up files created during setUp/test
        if os.path.exists(self.dummy_csv):
            os.remove(self.dummy_csv)
            
        for img_path in self.dummy_images:
            if os.path.exists(img_path):
                os.remove(img_path)
                
    def test_dataset_transforms_and_loading(self):
        """
        Verify AIDetectionDataset correctly preprocesses and returns images and labels.
        """
        transform = get_dinov2_transforms(img_size=224)
        dataset = AIDetectionDataset(csv_file=self.dummy_csv, transform=transform)
        
        self.assertEqual(len(dataset), 2)
        
        img, label = dataset[0]
        # Check shape is 3, 224, 224
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertEqual(label, 0)
        
        img2, label2 = dataset[1]
        self.assertEqual(label2, 1)

    def test_classification_heads(self):
        """
        Verify Linear and MLP classifier heads execute forward pass on dummy feature vectors.
        """
        embed_dim = 384
        batch_size = 4
        dummy_features = torch.randn(batch_size, embed_dim)
        
        # Test Linear Classifier
        linear_clf = LinearClassifier(embed_dim=embed_dim, num_classes=2)
        linear_output = linear_clf(dummy_features)
        self.assertEqual(linear_output.shape, (batch_size, 2))
        
        # Test MLP Classifier
        mlp_clf = MLPClassifier(embed_dim=embed_dim, num_classes=2, hidden_dim=128, dropout=0.2)
        mlp_output = mlp_clf(dummy_features)
        self.assertEqual(mlp_output.shape, (batch_size, 2))

    def test_end_to_end_classifier(self):
        """
        Verify end-to-end DINOv2Classifier handles raw image batch inputs successfully.
        """
        batch_size = 2
        dummy_img_batch = torch.randn(batch_size, 3, 224, 224)
        
        # Test DINOv2 end-to-end model (Linear head)
        model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type='linear', num_classes=2, freeze_backbone=True)
        outputs = model(dummy_img_batch)
        self.assertEqual(outputs.shape, (batch_size, 2))
        
        # Test DINOv2 end-to-end model (MLP head)
        model_mlp = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type='mlp', num_classes=2, freeze_backbone=True)
        outputs_mlp = model_mlp(dummy_img_batch)
        self.assertEqual(outputs_mlp.shape, (batch_size, 2))

    def test_faithfulness_perturbation(self):
        """
        Verify run_faithfulness_perturbation generates masked images and drop metrics.
        """
        model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type='mlp', num_classes=2, freeze_backbone=True)
        model.eval()
        dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        res = run_faithfulness_perturbation(dummy_img, model, device="cpu", threshold_pct=0.15)
        
        self.assertIn("orig_img_224", res)
        self.assertIn("attn_masked_img", res)
        self.assertIn("rand_masked_img", res)
        self.assertIn("base_confidence", res)
        self.assertIn("base_logit", res)
        self.assertIn("attribution_logit_drop", res)
        self.assertIn("random_logit_drop", res)
        self.assertIn("passed_logit_test", res)
        self.assertEqual(res["total_patches"], 256)
        self.assertEqual(res["num_patches_masked"], int(256 * 0.15))

    def test_llm_forensic_explanation(self):
        """
        Verify generate_forensic_explanation returns formatted forensic report.
        """
        dummy_img = Image.new("RGB", (224, 224), color=(50, 100, 150))
        res = generate_forensic_explanation(
            orig_img=dummy_img,
            prediction="AI_GENERATED",
            confidence=0.985,
            api_key=""  # offline test
        )
        self.assertIn("content", res)
        self.assertIn("status", res)
        self.assertIn("Lighting", res["content"])
        self.assertIn("Hair", res["content"])

if __name__ == "__main__":
    unittest.main()
