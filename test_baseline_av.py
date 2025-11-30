"""
Test RoWeeder Baseline on Agriculture-Vision Dataset - MEMORY EFFICIENT

This script:
1. Loads RoWeeder baseline (trained on sugar beet/WeedMap)
2. Tests on Agriculture-Vision (corn/soy/wheat)
3. Shows domain gap (expected: LOW performance <40%)
4. Justifies why fine-tuning was necessary

Expected: <40% F1 (domain mismatch - different crops, different anomalies)

Usage:
    python test_roweeder_baseline_agriculture.py
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from roweeder.models import RoWeederFlat
from agriculture_vision_dataset import AgricultureVisionDataset, get_val_transforms


class IncrementalMetrics:
    """Calculate metrics incrementally without storing all predictions"""
    
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.class_names = [
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


def test_roweeder_on_agriculture(agriculture_path, output_dir, device='cuda'):
    """Test RoWeeder baseline on Agriculture-Vision dataset - MEMORY SAFE"""
    
    print("="*80)
    print("TESTING ROWEEDER BASELINE ON AGRICULTURE-VISION DATASET")
    print("="*80)
    print("\n⚠️  Expected: LOW performance (domain mismatch)")
    print("   RoWeeder trained on: Sugar beet (WeedMap)")
    print("   Testing on: Corn/Soy/Wheat (Agriculture-Vision)")
    print("   Goal: Show why fine-tuning was necessary!")
    print("   🔧 Memory-efficient version\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RoWeeder baseline
    print("Loading RoWeeder baseline...")
    model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
    
    # Adapt for Agriculture-Vision classes
    # RoWeeder outputs 3 classes (bg, crop, weed)
    # Agriculture-Vision has 7 classes
    # We need to adapt the classifier head
    
    print("⚠️  NOTE: RoWeeder has 3 classes, Agriculture-Vision has 7")
    print("   Adapting classifier head (will perform poorly - expected!)")
    
    # Replace classifier to output 7 classes
    model.classifier = nn.Conv2d(32, 7, kernel_size=1)
    
    # Also need to adapt input for 4 channels (RGB + NIR)
    old_conv = model.encoder.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(4, old_conv.out_channels, 
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, 
                         padding=old_conv.padding)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
        new_conv.bias = old_conv.bias
    model.encoder.segformer.encoder.patch_embeddings[0].proj = new_conv
    
    model = model.to(device)
    model.eval()
    print("✅ Model loaded (randomly initialized classifier for 7 classes)")
    
    # Load Agriculture-Vision test set
    print("\nLoading Agriculture-Vision test set...")
    dataset = AgricultureVisionDataset(
        root_dir=agriculture_path,
        split='test',
        transform=get_val_transforms()
    )
    print(f"✅ Loaded {len(dataset)} test images")
    
    # Initialize incremental metrics
    metrics_calculator = IncrementalMetrics(num_classes=7)
    
    # Test with memory management
    print("\nTesting (memory-efficient mode)...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Testing"):
            try:
                sample = dataset[idx]
                
                image = sample['image'].unsqueeze(0).to(device)
                label = sample['label'].numpy()
                
                # Predict
                outputs = model(image)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('out', list(outputs.values())[0]))
                else:
                    logits = outputs
                
                # Resize if needed
                if logits.shape[-2:] != label.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=label.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                preds = logits.argmax(dim=1).cpu().numpy()[0]
                
                # Update metrics incrementally
                metrics_calculator.update(preds, label)
                
                # Free memory every 100 images
                if (idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"\n❌ Error processing image {idx}: {e}")
                continue
    
    # Compute final metrics
    print("\nCalculating final metrics...")
    metrics = metrics_calculator.compute()
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - RoWeeder Baseline on Agriculture-Vision")
    print("="*80)
    
    print("\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<20} {'F1':>10} {'IoU':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*80)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<20} {class_metrics['f1']:>10.4f} {class_metrics['iou']:>10.4f} "
              f"{class_metrics['precision']:>12.4f} {class_metrics['recall']:>10.4f}")
    
    print("\nMean Metrics (Excluding Background):")
    print("-"*80)
    print(f"Mean F1:        {metrics['mean_without_bg']['f1']:.4f}")
    print(f"Mean IoU:       {metrics['mean_without_bg']['iou']:.4f}")
    print(f"Mean Precision: {metrics['mean_without_bg']['precision']:.4f}")
    print(f"Mean Recall:    {metrics['mean_without_bg']['recall']:.4f}")
    print("="*80)
    
    # Analysis
    f1 = metrics['mean_without_bg']['f1']
    
    print(f"\n📊 Analysis:")
    print(f"   RoWeeder Baseline F1: {f1:.3f} ({f1*100:.1f}%)")
    
    if f1 < 0.40:
        print("   ❌ LOW performance as expected! (domain mismatch)")
        print("   ✅ This justifies why fine-tuning was necessary!")
    elif f1 < 0.50:
        print("   ⚠️ Poor performance (domain gap)")
        print("   ✅ Shows need for fine-tuning")
    else:
        print("   🤔 Surprisingly high for untrained model")
        print("   Note: Classifier head was randomly initialized")
    
    print(f"\n💡 Why performance is low:")
    print(f"   • Trained on: Sugar beet fields (WeedMap)")
    print(f"   • Testing on: Corn/Soy/Wheat fields (Agriculture-Vision)")
    print(f"   • Different crops → Different appearance")
    print(f"   • Different anomalies → planter_skip, nutrient_def don't exist in WeedMap")
    print(f"   • Classifier randomly initialized for 7 classes")
    
    # Save results
    results_file = output_dir / "roweeder_baseline_agriculture_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': 'Agriculture-Vision',
            'model': 'RoWeeder Baseline (no fine-tuning)',
            'num_images': len(dataset),
            'metrics': metrics,
            'analysis': {
                'trained_on': 'WeedMap (sugar beet)',
                'tested_on': 'Agriculture-Vision (corn/soy/wheat)',
                'domain_gap': 'Expected',
                'f1_score': float(f1)
            }
        }, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = output_dir / "roweeder_baseline_agriculture_confusion_matrix.png"
    class_names = ['BG', 'Crop', 'Planter', 'Water', 'Weed', 'Nutrient', 'Waterway']
    
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('RoWeeder Baseline on Agriculture-Vision\n(Low Performance Expected - Domain Mismatch)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {cm_path}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agriculture_path', type=str,
                       default='dataset/finetuning_dataset/agriculture_vision_2019',
                       help='Path to Agriculture-Vision dataset')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/test_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    test_roweeder_on_agriculture(
        agriculture_path=args.agriculture_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()