"""
Test Reproduced RoWeeder Checkpoint on Agriculture-Vision Dataset

This script:
1. Loads YOUR reproduced RoWeeder checkpoint
2. Tests on Agriculture-Vision test set (corn/soy/wheat)
3. Calculates comprehensive metrics (F1, IoU, Precision, Recall)
4. Generates confusion matrix
5. Shows domain transfer from WeedMap to Agriculture-Vision

Expected: Lower than WeedMap (domain gap), but should show your model's capability

Usage:
    python test_reproduced_agriculture_vision.py --checkpoint weedmap-inference/models/reproduced_roweeder/best_checkpoint.pth --agriculture_path dataset/finetuning_dataset/agriculture_vision_2019
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc

# Import the model builder
from roweeder.models import build_roweeder_flat


class AgricultureVisionDataset:
    """
    Agriculture-Vision dataset loader for testing
    Uses the same structure as agriculture_vision_dataset.py
    
    Classes (7):
    - 0: background
    - 1: crop (healthy)
    - 2: planter_skip
    - 3: water
    - 4: weed_cluster
    - 5: nutrient_deficiency
    - 6: waterway
    """
    
    def __init__(self, root_dir, split='test', image_size=512):
        """
        Args:
            root_dir: Path to Agriculture-Vision dataset (agriculture_vision_2019 folder)
            split: 'train', 'val', or 'test'
            image_size: Image size (default 512)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        # Class mapping
        self.classes = [
            'background', 'crop', 'planter_skip', 'water',
            'weed_cluster', 'nutrient_deficiency', 'waterway'
        ]
        self.num_classes = len(self.classes)
        
        # Anomaly types
        self.anomaly_types = [
            'planter_skip',
            'water', 
            'weed_cluster',
            'nutrient_deficiency',
            'waterway'
        ]
        
        # Load split information
        self.image_ids = self._load_split()
        print(f"Found {len(self.image_ids)} {split} images")
    
    def _load_split(self):
        """Load image IDs for the specified split"""
        # Load split counts
        split_file = self.root_dir / 'split_stats.json'
        
        if not split_file.exists():
            print(f"❌ Error: split_stats.json not found at {split_file}")
            return []
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Get all image IDs from RGB folder
        rgb_dir = self.root_dir / 'field_images' / 'rgb'
        
        if not rgb_dir.exists():
            print(f"❌ Error: RGB directory not found at {rgb_dir}")
            return []
        
        all_image_files = sorted([f for f in rgb_dir.glob('*.jpg')])
        all_image_ids = [f.stem for f in all_image_files]
        
        print(f"Total images found: {len(all_image_ids)}")
        
        # Split images based on counts from split_stats.json
        train_count = split_data['train']['image_count']
        val_count = split_data['val']['image_count']
        test_count = split_data['test']['image_count']
        
        total_expected = train_count + val_count + test_count
        print(f"Expected total: {total_expected} (train: {train_count}, val: {val_count}, test: {test_count})")
        
        if len(all_image_ids) != total_expected:
            print(f"⚠️  Warning: Found {len(all_image_ids)} images but expected {total_expected}")
        
        # Split the images
        if self.split == 'train':
            return all_image_ids[:train_count]
        elif self.split == 'val':
            return all_image_ids[train_count:train_count + val_count]
        else:  # test
            return all_image_ids[train_count + val_count:]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Load image and mask"""
        from PIL import Image
        
        image_id = self.image_ids[idx]
        
        # Load RGB image
        rgb_path = self.root_dir / 'field_images' / 'rgb' / f'{image_id}.jpg'
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb_img = np.array(rgb_img)
        
        # Load NIR image
        nir_path = self.root_dir / 'field_images' / 'nir' / f'{image_id}.jpg'
        nir_img = Image.open(nir_path).convert('L')
        nir_img = nir_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        nir_img = np.array(nir_img)
        
        # For RoWeeder Flat, we only use RGB (3 channels)
        # NOTE: If your model uses 4 channels (RGBN), uncomment the line below
        image = rgb_img  # (H, W, 3)
        # image = np.dstack([rgb_img, nir_img[:, :, np.newaxis]])  # (H, W, 4) - for RGBN
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Load valid pixel mask
        mask_path = self.root_dir / 'field_masks' / f'{image_id}.png'
        if mask_path.exists():
            valid_mask = Image.open(mask_path).convert('L')
            valid_mask = valid_mask.resize((self.image_size, self.image_size), Image.NEAREST)
            valid_mask = np.array(valid_mask)
            valid_mask = (valid_mask > 0).astype(np.uint8)
        else:
            # If no mask, assume all pixels are valid
            valid_mask = np.ones((self.image_size, self.image_size), dtype=np.uint8)
        
        # Initialize label mask (all pixels start as crop=1)
        label = np.ones((self.image_size, self.image_size), dtype=np.uint8)
        
        # Set invalid pixels to background (0)
        label[valid_mask == 0] = 0
        
        # Load anomaly masks and assign class IDs
        for idx_anomaly, anomaly_type in enumerate(self.anomaly_types):
            anomaly_path = self.root_dir / 'field_labels' / anomaly_type / f'{image_id}.png'
            
            if anomaly_path.exists():
                anomaly_mask = Image.open(anomaly_path).convert('L')
                anomaly_mask = anomaly_mask.resize((self.image_size, self.image_size), Image.NEAREST)
                anomaly_mask = np.array(anomaly_mask)
                anomaly_mask = (anomaly_mask > 0).astype(np.uint8)
                
                # Assign class ID (2-6 for anomalies)
                label[anomaly_mask > 0] = idx_anomaly + 2
        
        return {
            'image': image,
            'label': label,
            'id': image_id
        }


class IncrementalMetrics:
    """Calculate metrics incrementally without storing all predictions"""
    
    def __init__(self, num_classes=7, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'background', 'crop', 'planter_skip', 'water',
            'weed_cluster', 'nutrient_deficiency', 'waterway'
        ]
        
        # Confusion matrix accumulator
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, predictions, labels):
        """Update metrics with new batch"""
        preds_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Update confusion matrix
        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                mask = (labels_flat == true_class) & (preds_flat == pred_class)
                self.confusion_matrix[true_class, pred_class] += mask.sum()
    
    def compute(self):
        """Compute final metrics from confusion matrix"""
        metrics = {
            'per_class': {},
            'mean_without_bg': {},
            'confusion_matrix': self.confusion_matrix.tolist()
        }
        
        f1_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        
        for cls in range(self.num_classes):
            # Get TP, FP, FN from confusion matrix
            tp = self.confusion_matrix[cls, cls]
            fp = self.confusion_matrix[:, cls].sum() - tp
            fn = self.confusion_matrix[cls, :].sum() - tp
            
            # Calculate metrics
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            iou = tp / (tp + fp + fn + 1e-7)
            
            metrics['per_class'][self.class_names[cls]] = {
                'f1': float(f1),
                'iou': float(iou),
                'precision': float(precision),
                'recall': float(recall),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
            
            # Skip background for mean
            if cls > 0:
                f1_scores.append(f1)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)
        
        # Mean metrics (excluding background)
        metrics['mean_without_bg'] = {
            'f1': float(np.mean(f1_scores)),
            'iou': float(np.mean(iou_scores)),
            'precision': float(np.mean(precision_scores)),
            'recall': float(np.mean(recall_scores))
        }
        
        return metrics


def load_reproduced_checkpoint(checkpoint_path, num_classes, device):
    """
    Load YOUR reproduced RoWeeder checkpoint and adapt for Agriculture-Vision
    """
    print(f"\n{'='*80}")
    print("LOADING REPRODUCED CHECKPOINT")
    print(f"{'='*80}")
    print(f"Path: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"\n✅ Checkpoint Info:")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Fold: {checkpoint.get('fold', 'unknown')}")
    print(f"   Validation F1: {checkpoint.get('best_f1', checkpoint.get('val_f1', 'unknown'))}")
    print(f"   Test F1: {checkpoint.get('test_f1', 'unknown')}")
    
    # Build the model using the correct builder function
    print(f"\n🏗️  Building RoWeeder Flat model...")
    print(f"   Original classes: 3 (background, crop, weed)")
    print(f"   Target classes: {num_classes} (Agriculture-Vision)")
    
    # Build model with original architecture
    model = build_roweeder_flat(
        input_channels=['R', 'G', 'B'],
        version="nvidia/mit-b0",
        fusion='concat',
        upsampling='interpolate',
        spatial_conv=True,
        blocks=4
    )
    print("✅ Model created successfully!")
    
    # Load state dict (from WeedMap training)
    print(f"\n📦 Loading checkpoint weights...")
    state_dict = checkpoint['model_state_dict']
    
    # Strip 'model.' prefix if present
    first_key = list(state_dict.keys())[0]
    if first_key.startswith('model.'):
        print("   Detected 'model.' prefix in checkpoint keys, removing it...")
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        print("✅ Stripped 'model.' prefix from all keys")
    
    # Load weights (strict mode for encoder, ignore classifier)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("✅ Loaded state dict perfectly!")
    else:
        if len(unexpected_keys) > 0:
            print(f"   ⚠️  Unexpected keys (will be ignored): {len(unexpected_keys)}")
        if len(missing_keys) > 0:
            print(f"   ⚠️  Missing keys (will be randomly initialized): {len(missing_keys)}")
    
    # Adapt classifier for Agriculture-Vision (7 classes)
    print(f"\n🔧 Adapting classifier for Agriculture-Vision...")
    print(f"   Original: 3 classes → Agriculture-Vision: {num_classes} classes")
    
    # Replace classifier head
    # Find the classifier layer
    if hasattr(model, 'classifier'):
        old_classifier = model.classifier
        if isinstance(old_classifier, nn.Conv2d):
            new_classifier = nn.Conv2d(
                old_classifier.in_channels,
                num_classes,
                kernel_size=old_classifier.kernel_size,
                stride=old_classifier.stride,
                padding=old_classifier.padding
            )
            # Initialize with small random weights
            nn.init.xavier_uniform_(new_classifier.weight)
            nn.init.zeros_(new_classifier.bias)
            model.classifier = new_classifier
            print(f"✅ Classifier adapted: {old_classifier.in_channels} → {num_classes} classes")
        else:
            print(f"⚠️  Classifier is not Conv2d, type: {type(old_classifier)}")
    else:
        print("⚠️  No 'classifier' attribute found in model")
    
    model = model.to(device)
    model.eval()
    
    print(f"\n✅ Model loaded and adapted for Agriculture-Vision!")
    return model


def test_reproduced_on_agriculture(checkpoint_path, agriculture_path, output_dir, device='cuda'):
    """Test reproduced RoWeeder checkpoint on Agriculture-Vision"""
    
    print("="*80)
    print("TESTING REPRODUCED ROWEEDER ON AGRICULTURE-VISION DATASET")
    print("="*80)
    print("\n📊 Testing Domain Transfer:")
    print("   Trained on: WeedMap (sugar beet)")
    print("   Testing on: Agriculture-Vision (corn/soy/wheat)")
    print("   Classifier: Adapted for 7 classes (randomly initialized)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model = load_reproduced_checkpoint(checkpoint_path, num_classes=7, device=device)
    
    # Load dataset
    print(f"\n{'='*80}")
    print("LOADING AGRICULTURE-VISION TEST SET")
    print(f"{'='*80}")
    dataset = AgricultureVisionDataset(
        root_dir=agriculture_path,
        split='test',
        image_size=512
    )
    
    if len(dataset) == 0:
        print("❌ No test images found! Check dataset path.")
        return
    
    # Initialize incremental metrics
    metrics_calculator = IncrementalMetrics(
        num_classes=7,
        class_names=['background', 'crop', 'planter_skip', 'water',
                     'weed_cluster', 'nutrient_deficiency', 'waterway']
    )
    
    # Test
    print(f"\n{'='*80}")
    print(f"TESTING ON {len(dataset)} IMAGES")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Testing"):
            try:
                sample = dataset[idx]
                
                # Prepare image (H, W, C) -> (1, C, H, W)
                image = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0).float()
                image = image.to(device)
                
                # Predict
                outputs = model(image)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('out', list(outputs.values())[0]))
                else:
                    logits = outputs
                
                # Upsample to 512x512 if needed
                if logits.shape[-2:] != (512, 512):
                    logits = nn.functional.interpolate(
                        logits,
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Get predictions
                preds = logits.argmax(dim=1).cpu().numpy()[0]
                
                # Update metrics incrementally
                metrics_calculator.update(preds, sample['label'])
                
                # Free memory every 100 images
                if (idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"\n❌ Error processing image {idx}: {e}")
                continue
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("CALCULATING METRICS")
    print(f"{'='*80}")
    
    metrics = metrics_calculator.compute()
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<25} {'F1':>10} {'IoU':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*80)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<25} {class_metrics['f1']:>10.4f} {class_metrics['iou']:>10.4f} "
              f"{class_metrics['precision']:>12.4f} {class_metrics['recall']:>10.4f}")
    
    print("\nMean Metrics (Excluding Background):")
    print("-"*80)
    print(f"Mean F1:        {metrics['mean_without_bg']['f1']:.4f} ({metrics['mean_without_bg']['f1']*100:.2f}%)")
    print(f"Mean IoU:       {metrics['mean_without_bg']['iou']:.4f} ({metrics['mean_without_bg']['iou']*100:.2f}%)")
    print(f"Mean Precision: {metrics['mean_without_bg']['precision']:.4f} ({metrics['mean_without_bg']['precision']*100:.2f}%)")
    print(f"Mean Recall:    {metrics['mean_without_bg']['recall']:.4f} ({metrics['mean_without_bg']['recall']*100:.2f}%)")
    print("="*80)
    
    # Analysis
    our_f1 = metrics['mean_without_bg']['f1']
    
    print(f"\n📊 Analysis:")
    print(f"   Reproduced Model F1: {our_f1:.3f} ({our_f1*100:.1f}%)")
    
    if our_f1 < 0.30:
        print("   ⚠️  LOW performance (expected due to domain gap + random classifier)")
        print("   Note: Classifier randomly initialized for new classes")
    elif our_f1 < 0.50:
        print("   📊 Moderate performance considering domain gap")
        print("   Note: Encoder learned from WeedMap, classifier random")
    else:
        print("   🎉 Good performance despite domain transfer!")
    
    print(f"\n💡 Understanding the results:")
    print(f"   • Encoder: Pre-trained on WeedMap (should transfer some features)")
    print(f"   • Classifier: Randomly initialized for 7 classes (not trained!)")
    print(f"   • Domain gap: Sugar beet → Corn/Soy/Wheat")
    print(f"   • New classes: planter_skip, nutrient_def, etc. (unseen during training)")
    print(f"   • This shows the baseline - fine-tuning should improve this!")
    
    # Save results
    results_file = output_dir / "reproduced_roweeder_agriculture_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': 'Agriculture-Vision',
            'model': 'Reproduced RoWeeder (WeedMap checkpoint, adapted classifier)',
            'checkpoint': str(checkpoint_path),
            'num_images': len(dataset),
            'metrics': metrics,
            'analysis': {
                'trained_on': 'WeedMap (sugar beet)',
                'tested_on': 'Agriculture-Vision (corn/soy/wheat)',
                'classifier_status': 'randomly initialized for 7 classes',
                'f1_score': float(our_f1)
            }
        }, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Generate confusion matrix
    cm_path = output_dir / "reproduced_roweeder_agriculture_confusion_matrix.png"
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['BG', 'Crop', 'Planter', 'Water', 'Weed', 'Nutrient', 'Waterway'],
        save_path=cm_path
    )
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """Generate and save confusion matrix"""
    cm_array = np.array(cm)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Reproduced RoWeeder on Agriculture-Vision\n(Domain Transfer: WeedMap → Agriculture-Vision)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Reproduced RoWeeder on Agriculture-Vision')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to reproduced checkpoint (e.g., best_checkpoint.pth)')
    parser.add_argument('--agriculture_path', type=str, required=True,
                       help='Path to Agriculture-Vision dataset root')
    parser.add_argument('--output_dir', type=str, default='results/reproduced_agriculture',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run test
    test_reproduced_on_agriculture(
        checkpoint_path=args.checkpoint,
        agriculture_path=args.agriculture_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()