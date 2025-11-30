"""
Test Phase 1 Finetuned Model on WeedMap Dataset - FINAL CORRECTED VERSION

This script:
1. Loads your Phase 1 finetuned model (trained on Agriculture-Vision)
2. Tests on WeedMap dataset (sugar beet) with proper labels
3. Shows domain specificity - model optimized for Agriculture-Vision

Expected: <60% F1 (domain mismatch - different crop type)

Usage:
    python test_phase1_weedmap.py --weedmap_path dataset/test
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

from roweeder.models import RoWeederFlat


class WeedMapDataset:
    """WeedMap dataset loader - CORRECTED for actual label format"""
    
    def __init__(self, root_dir, field_ids=['003', '004'], image_size=512):
        self.root_dir = Path(root_dir)
        self.field_ids = field_ids
        self.image_size = image_size
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} test images with labels from fields: {field_ids}")
    
    def _find_samples(self):
        samples = []
        
        for field_id in self.field_ids:
            field_path = self.root_dir / "RedEdge" / field_id
            
            if not field_path.exists():
                print(f"Warning: Field {field_id} not found at {field_path}")
                continue
            
            gt_path = field_path / "groundtruth"
            
            if not gt_path.exists():
                print(f"Warning: Groundtruth folder not found at {gt_path}")
                continue
            
            # Find all GroundTruth_iMap files
            gt_files = sorted(gt_path.glob("*_GroundTruth_iMap.png"))
            
            for gt_file in gt_files:
                # Extract frame ID
                parts = gt_file.stem.split('_')
                frame_num = parts[1].replace('frame', '')
                
                # Check if this groundtruth has labels
                gt_check = np.array(Image.open(gt_file))
                unique_vals = np.unique(gt_check)
                
                # Skip if only zeros
                if len(unique_vals) == 1 and unique_vals[0] == 0:
                    continue
                
                # Build tile paths
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
        sample = self.samples[idx]
        
        # Load channels and resize to 512×512
        channels = {}
        for name, path in sample['channels'].items():
            img = Image.open(path).resize((512, 512), Image.BILINEAR)
            channels[name] = np.array(img)
        
        # Compose RGB + NIR (4 channels - Phase 1 uses NIR)
        image = np.stack([
            channels['red'],
            channels['green'],
            channels['blue'],
            channels['nir']
        ], axis=-1)
        
        # Load groundtruth and resize
        gt = np.array(Image.open(sample['groundtruth']).resize((512, 512), Image.NEAREST))
        
        # WeedMap original labels (in iMap):
        # 0 = crop, 2 = background, 10000 = weed
        
        # Store original in WeedMap format for metrics
        # Map to WeedMap 3-class format first:
        # 0=crop, 1=weed, 2=background
        original_label = np.zeros_like(gt, dtype=np.uint8)
        original_label[gt == 0] = 0      # crop → 0
        original_label[gt == 10000] = 1  # weed → 1
        original_label[gt == 2] = 2      # background → 2
        
        # Phase 1 uses 7 classes, map WeedMap to closest:
        # WeedMap crop(0) → Ag-Vision crop(1)
        # WeedMap weed(1) → Ag-Vision weed_cluster(4)
        # WeedMap bg(2) → Ag-Vision background(0)
        label = np.zeros_like(gt, dtype=np.uint8)
        label[gt == 0] = 1      # crop → 1
        label[gt == 10000] = 4  # weed → 4 (weed_cluster)
        label[gt == 2] = 0      # background → 0
        
        return {
            'image': image.astype(np.float32) / 255.0,
            'label': label,
            'original_label': original_label,
            'field_id': sample['field_id'],
            'frame_id': sample['frame_id']
        }


def calculate_metrics_weedmap_adapted(predictions, original_labels):
    """
    Calculate metrics for WeedMap using Phase 1 model
    
    Phase 1 predicts 7 classes, WeedMap has 3
    Map predictions back to WeedMap classes for fair comparison
    """
    
    # Map Phase 1 predictions to WeedMap classes
    # Phase 1: 0=bg, 1=crop, 2=planter_skip, 3=water, 4=weed_cluster, 5=nutrient_def, 6=waterway
    # WeedMap: 0=crop, 1=weed, 2=background
    
    mapped_preds = np.zeros_like(predictions)
    mapped_preds[predictions == 0] = 2  # bg → bg
    mapped_preds[predictions == 1] = 0  # crop → crop
    mapped_preds[predictions == 4] = 1  # weed_cluster → weed
    # All other classes → background
    mapped_preds[(predictions != 0) & (predictions != 1) & (predictions != 4)] = 2
    
    class_names = ['crop', 'weed', 'background']
    metrics = {'per_class': {}, 'mean': {}}
    
    preds_flat = mapped_preds.flatten()
    labels_flat = original_labels.flatten()
    
    f1_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    
    for cls in range(3):
        pred_mask = (preds_flat == cls)
        label_mask = (labels_flat == cls)
        
        tp = np.logical_and(pred_mask, label_mask).sum()
        fp = np.logical_and(pred_mask, ~label_mask).sum()
        fn = np.logical_and(~pred_mask, label_mask).sum()
        
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
            'fn': int(fn)
        }
        
        # Skip background for mean
        if cls < 2:
            f1_scores.append(f1)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    metrics['mean'] = {
        'f1': float(np.mean(f1_scores)),
        'iou': float(np.mean(iou_scores)),
        'precision': float(np.mean(precision_scores)),
        'recall': float(np.mean(recall_scores))
    }
    
    return metrics


def test_phase1_on_weedmap(checkpoint_path, weedmap_path, output_dir, device='cuda'):
    """Test Phase 1 finetuned model on WeedMap"""
    
    print("="*80)
    print("TESTING PHASE 1 FINETUNED MODEL ON WEEDMAP")
    print("="*80)
    print("\n⚠️  Expected: <60% F1 (domain mismatch)")
    print("   Model trained on: Corn/Soy/Wheat (Agriculture-Vision)")
    print("   Testing on: Sugar beet (WeedMap)")
    print("   Goal: Show domain specificity!\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Phase 1 checkpoint
    print("Loading Phase 1 finetuned model...")
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        possible_paths = [
            Path("outputs/agriculture_vision_v2/best_model.pth"),
            Path("outputs/agriculture_vision_v2/checkpoint_epoch_24.pth"),
        ]
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        else:
            raise FileNotFoundError("No Phase 1 checkpoint found!")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model
    model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
    
    # Adapt for Agriculture-Vision (7 classes, 4 channels)
    model.classifier = nn.Conv2d(32, 7, kernel_size=1)
    
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
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded Phase 1 checkpoint (epoch {epoch})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded Phase 1 checkpoint")
    
    model = model.to(device)
    model.eval()
    
    # Load WeedMap test set
    print("\nLoading WeedMap test set...")
    dataset = WeedMapDataset(weedmap_path, field_ids=['003', '004'])
    
    if len(dataset) == 0:
        print("❌ No test images found!")
        return
    
    # Test
    print(f"\nTesting on {len(dataset)} images...")
    all_predictions = []
    all_original_labels = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Testing"):
            sample = dataset[idx]
            
            image = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0)
            image = image.to(device)
            
            outputs = model(image)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('out', list(outputs.values())[0]))
            else:
                logits = outputs
            
            preds = logits.argmax(dim=1).cpu().numpy()[0]
            
            all_predictions.append(preds)
            all_original_labels.append(sample['original_label'])
    
    all_predictions = np.array(all_predictions)
    all_original_labels = np.array(all_original_labels)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics_weedmap_adapted(all_predictions, all_original_labels)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - Phase 1 Finetuned on WeedMap")
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
    
    # Analysis
    baseline_f1 = 0.753  # RoWeeder baseline
    our_f1 = metrics['mean']['f1']
    diff = our_f1 - baseline_f1
    
    print(f"\n📊 Comparison:")
    print(f"   RoWeeder Baseline (WeedMap): {baseline_f1:.3f} (75.3%)")
    print(f"   Phase 1 Finetuned (Ag-Vis):  {our_f1:.3f} ({our_f1*100:.1f}%)")
    print(f"   Difference:                  {diff:+.3f} ({diff*100:+.1f}%)")
    
    if our_f1 < baseline_f1:
        print(f"\n✅ Expected result: Phase 1 performs worse on WeedMap")
        print(f"   • Trained on: Corn/Soy/Wheat")
        print(f"   • Testing on: Sugar beet (different crop!)")
        print(f"   • Shows model is specialized for Agriculture-Vision")
    else:
        print(f"\n🤔 Surprisingly, Phase 1 matches/beats baseline")
        print(f"   • Model may have learned general crop/weed features")
    
    # Save results
    results_file = output_dir / "phase1_finetuned_weedmap_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': 'WeedMap',
            'model': 'Phase 1 Finetuned (Agriculture-Vision)',
            'checkpoint': str(checkpoint_path),
            'num_images': len(dataset),
            'metrics': metrics,
            'baseline_f1': baseline_f1,
            'our_f1': float(our_f1),
            'difference': float(diff)
        }, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Confusion matrix
    # Map predictions for visualization
    mapped_preds = np.zeros_like(all_predictions)
    mapped_preds[all_predictions == 0] = 2
    mapped_preds[all_predictions == 1] = 0
    mapped_preds[all_predictions == 4] = 1
    mapped_preds[(all_predictions != 0) & (all_predictions != 1) & (all_predictions != 4)] = 2
    
    cm_path = output_dir / "phase1_finetuned_weedmap_confusion_matrix.png"
    cm = confusion_matrix(all_original_labels.flatten(), mapped_preds.flatten())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Crop', 'Weed', 'Background'],
                yticklabels=['Crop', 'Weed', 'Background'])
    plt.title(f'Phase 1 Finetuned on WeedMap\nF1 Score: {our_f1:.3f} ({our_f1*100:.1f}%)')
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
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/agriculture_vision_v2/best_model.pth')
    parser.add_argument('--weedmap_path', type=str, required=True,
                       help='Path to WeedMap dataset root (e.g., dataset/test)')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/test_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    test_phase1_on_weedmap(
        checkpoint_path=args.checkpoint,
        weedmap_path=args.weedmap_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()