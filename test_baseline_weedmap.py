"""
Test RoWeeder Baseline on WeedMap Dataset - FINAL CORRECTED VERSION

This script:
1. Loads RoWeeder baseline from HuggingFace
2. Tests on WeedMap test set with proper groundtruth labels
3. Calculates comprehensive metrics (F1, IoU, Precision, Recall)
4. Generates confusion matrix
5. Saves detailed results

Expected: ~75% F1 (verify their paper claim)

Usage:
    python test_roweeder_baseline_weedmap.py --weedmap_path dataset/test
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from roweeder.models import RoWeederFlat


class WeedMapDataset:
    """
    WeedMap dataset loader - CORRECTED for actual label format
    
    Label files: *_GroundTruth_iMap.png
    Label values: 0=crop, 2=background, 10000=weed
    """
    
    def __init__(self, root_dir, field_ids=['003', '004'], image_size=512):
        """
        Args:
            root_dir: Path to WeedMap dataset (e.g., dataset/test)
            field_ids: Which fields to use for testing
            image_size: Image size (default 512)
        """
        self.root_dir = Path(root_dir)
        self.field_ids = field_ids
        self.image_size = image_size
        
        # Find all images with valid labels
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} test images with labels from fields: {field_ids}")
    
    def _find_samples(self):
        """Find all valid image/label pairs"""
        samples = []
        
        for field_id in self.field_ids:
            field_path = self.root_dir / "RedEdge" / field_id
            
            if not field_path.exists():
                print(f"Warning: Field {field_id} not found at {field_path}")
                continue
            
            # Ground truth files (iMap files contain actual labels)
            gt_path = field_path / "groundtruth"
            
            if not gt_path.exists():
                print(f"Warning: Groundtruth folder not found at {gt_path}")
                continue
            
            # Find all GroundTruth_iMap files
            gt_files = sorted(gt_path.glob("*_GroundTruth_iMap.png"))
            
            for gt_file in gt_files:
                # Extract frame ID from filename
                # Format: 000_frame0008_GroundTruth_iMap.png
                parts = gt_file.stem.split('_')
                frame_num = parts[1].replace('frame', '')  # "0008"
                
                # Check if this groundtruth actually has labels (not all zeros)
                gt_check = np.array(Image.open(gt_file))
                unique_vals = np.unique(gt_check)
                
                # Skip if only zeros (no labels)
                if len(unique_vals) == 1 and unique_vals[0] == 0:
                    continue
                
                # Build tile image paths
                # Note: tile files don't have field prefix, just frame number
                tile_path = field_path / "tile"
                frame_name = f"frame{frame_num}"
                
                channels = {
                    'blue': tile_path / "B" / f"{frame_name}.png",
                    'green': tile_path / "G" / f"{frame_name}.png",
                    'red': tile_path / "R" / f"{frame_name}.png",
                    'nir': tile_path / "NIR" / f"{frame_name}.png",
                    'rededge': tile_path / "RE" / f"{frame_name}.png",
                }
                
                # Verify all files exist
                if all(path.exists() for path in channels.values()):
                    samples.append({
                        'field_id': field_id,
                        'frame_id': frame_name,
                        'channels': channels,
                        'groundtruth': gt_file
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load image and mask"""
        sample = self.samples[idx]
        
        # Load all channels
        channels = {}
        for name, path in sample['channels'].items():
            img = Image.open(path).resize((512, 512), Image.BILINEAR)
            channels[name] = np.array(img)
        
        # Compose RGB + NIR (4 channels for RoWeeder)
        image = np.stack([
            channels['red'],
            channels['green'],
            channels['blue']
        ], axis=-1)  # (H, W, 4)
        
        # Load groundtruth
        gt = np.array(Image.open(sample['groundtruth']).resize((512, 512), Image.NEAREST))
        
        # WeedMap label encoding (in iMap files):
        # 0 = crop (sugar beet)
        # 2 = background/soil
        # 10000 = weed
        
        # Remap to RoWeeder output format:
        # 0 = background
        # 1 = crop
        # 2 = weed
        
        label = np.zeros_like(gt, dtype=np.uint8)
        label[gt == 0] = 1      # crop → 1
        label[gt == 2] = 0      # background → 0
        label[gt == 10000] = 2  # weed → 2
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        return {
            'image': image,
            'label': label,
            'field_id': sample['field_id'],
            'frame_id': sample['frame_id']
        }


def calculate_metrics(predictions, labels, num_classes=3, class_names=['background', 'crop', 'weed']):
    """Calculate comprehensive metrics"""
    metrics = {
        'per_class': {},
        'mean': {}
    }
    
    # Flatten arrays
    preds_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    # Per-class metrics
    f1_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    
    for cls in range(num_classes):
        pred_mask = (preds_flat == cls)
        label_mask = (labels_flat == cls)
        
        tp = np.logical_and(pred_mask, label_mask).sum()
        fp = np.logical_and(pred_mask, ~label_mask).sum()
        fn = np.logical_and(~pred_mask, label_mask).sum()
        tn = np.logical_and(~pred_mask, ~label_mask).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        
        metrics['per_class'][class_names[cls]] = {
            'f1': float(f1),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
        
        # Only include crop and weed in mean (skip background)
        if cls > 0:
            f1_scores.append(f1)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    # Mean metrics (excluding background)
    metrics['mean'] = {
        'f1': float(np.mean(f1_scores)),
        'iou': float(np.mean(iou_scores)),
        'precision': float(np.mean(precision_scores)),
        'recall': float(np.mean(recall_scores))
    }
    
    return metrics


def plot_confusion_matrix(predictions, labels, class_names, save_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(labels.flatten(), predictions.flatten())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - RoWeeder on WeedMap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def test_roweeder_baseline(weedmap_path, output_dir, device='cuda'):
    """Test RoWeeder baseline on WeedMap"""
    
    print("="*80)
    print("TESTING ROWEEDER BASELINE ON WEEDMAP DATASET")
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading RoWeeder baseline from HuggingFace...")
    model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Load dataset
    print("\nLoading WeedMap test set...")
    dataset = WeedMapDataset(weedmap_path, field_ids=['002','004'])
    
    if len(dataset) == 0:
        print("❌ No test images found! Check dataset path.")
        return
    
    # Test
    print(f"\nTesting on {len(dataset)} images...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Testing"):
            sample = dataset[idx]
            
            # Prepare image
            image = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0)
            image = image.to(device)
            
            # Predict
            outputs = model(image)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('out', list(outputs.values())[0]))
            else:
                logits = outputs
            
            # Get predictions
            preds = logits.argmax(dim=1).cpu().numpy()[0]
            
            # Store
            all_predictions.append(preds)
            all_labels.append(sample['label'])
    
    # Concatenate
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(
        all_predictions,
        all_labels,
        num_classes=3,
        class_names=['background', 'crop', 'weed']
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<15} {'F1':>10} {'IoU':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*80)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<15} {class_metrics['f1']:>10.4f} {class_metrics['iou']:>10.4f} "
              f"{class_metrics['precision']:>12.4f} {class_metrics['recall']:>10.4f}")
    
    print("\nMean Metrics (Crop + Weed):")
    print("-"*80)
    print(f"Mean F1:        {metrics['mean']['f1']:.4f}")
    print(f"Mean IoU:       {metrics['mean']['iou']:.4f}")
    print(f"Mean Precision: {metrics['mean']['precision']:.4f}")
    print(f"Mean Recall:    {metrics['mean']['recall']:.4f}")
    print("="*80)
    
    # Compare with paper
    paper_f1 = 0.753
    our_f1 = metrics['mean']['f1']
    diff = our_f1 - paper_f1
    
    print(f"\n📊 Comparison with Paper:")
    print(f"   Paper reported: {paper_f1:.3f} (75.3%)")
    print(f"   Our result:     {our_f1:.3f} ({our_f1*100:.1f}%)")
    print(f"   Difference:     {diff:+.3f} ({diff*100:+.1f}%)")
    
    if abs(diff) < 0.02:
        print("   ✅ Results match paper! (within 2%)")
    elif abs(diff) < 0.05:
        print("   ✅ Close to paper (within 5%)")
    else:
        print(f"   ⚠️ Results differ by {abs(diff)*100:.1f}%")
    
    # Save results
    results_file = output_dir / "roweeder_baseline_weedmap_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': 'WeedMap',
            'model': 'RoWeeder Baseline',
            'num_images': len(dataset),
            'metrics': metrics,
            'paper_f1': paper_f1,
            'our_f1': our_f1,
            'difference': float(diff)
        }, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Generate confusion matrix
    cm_path = output_dir / "roweeder_baseline_weedmap_confusion_matrix.png"
    plot_confusion_matrix(
        all_predictions,
        all_labels,
        class_names=['Background', 'Crop', 'Weed'],
        save_path=cm_path
    )
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test RoWeeder Baseline on WeedMap')
    parser.add_argument('--weedmap_path', type=str, required=True,
                       help='Path to WeedMap dataset root (e.g., dataset/test)')
    parser.add_argument('--output_dir', type=str, default='outputs/test_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run test
    test_roweeder_baseline(
        weedmap_path=args.weedmap_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()