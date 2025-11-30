"""
Diagnostic script to understand the Agriculture-Vision training issues
Run this BEFORE starting training to identify problems
"""
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming these imports work in your environment
from agriculture_vision_dataset import (
    AgricultureVisionDataset,
    get_train_transforms,
    get_val_transforms
)


def analyze_dataset_distribution(dataset, name="Dataset", sample_size=None):
    """Analyze class distribution in dataset"""
    print(f"\n{'='*80}")
    print(f"Analyzing {name} Distribution")
    print(f"{'='*80}")
    
    class_counts = Counter()
    sample_size = sample_size or len(dataset)
    
    print(f"Sampling {sample_size} images...")
    for i in tqdm(range(min(sample_size, len(dataset)))):
        sample = dataset[i]
        label = sample['label']
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        unique, counts = np.unique(label, return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[int(cls)] += int(count)
    
    total_pixels = sum(class_counts.values())
    
    print(f"\nClass distribution:")
    print(f"{'Class':<5} {'Name':<25} {'Pixel Count':<15} {'Percentage':<12} {'Inverse Weight'}")
    print("-" * 80)
    
    class_names = [
        'background', 'crop', 'planter_skip', 'water', 
        'weed_cluster', 'nutrient_deficiency', 'waterway'
    ]
    
    weights = []
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        # Inverse frequency weight
        weight = total_pixels / (count * len(class_counts))
        weights.append(weight)
        
        cls_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        print(f"{cls:<5} {cls_name:<25} {count:<15,} {percentage:>10.2f}% {weight:>10.4f}")
    
    print("-" * 80)
    print(f"Total pixels analyzed: {total_pixels:,}")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(v for k, v in class_counts.items() if k != 0)  # Exclude background
    imbalance_ratio = max_count / min_count
    print(f"\n⚠️ Class imbalance ratio (max/min, excluding background): {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 100:
        print("⚠️ SEVERE CLASS IMBALANCE DETECTED!")
        print("   Recommendation: Use class weights and/or focal loss")
    
    return class_counts, weights


def test_model_output_shape(dataset_path):
    """Test if model output matches expected shape"""
    print(f"\n{'='*80}")
    print("Testing Model Output Shape")
    print(f"{'='*80}")
    
    try:
        from roweeder.models import RoWeederFlat
        
        print("Loading RoWeeder model...")
        model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
        model.eval()
        
        # Create dummy 4-channel input
        dummy_input = torch.randn(1, 3, 512, 512)  # Still 3 channels for now
        
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Extract logits
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, dict):
            logits = None
            for key in ('logits', 'preds', 'out', 'segmentation'):
                if key in output:
                    logits = output[key]
                    break
            if logits is None:
                logits = list(output.values())[0]
        
        print(f"Output shape: {logits.shape}")
        print(f"Output spatial size: {logits.shape[-2:]} (H, W)")
        print(f"Number of output channels: {logits.shape[1]}")
        
        if logits.shape[-2:] != (512, 512):
            print("\n⚠️ WARNING: Output resolution doesn't match input!")
            print(f"   Output: {logits.shape[-2:]} vs Input: (512, 512)")
            print(f"   Downsampling factor: {512 / logits.shape[-2]:.1f}x")
            print("\n   ✅ Solution: The training script must UPSAMPLE the output to match labels!")
            print("      This is now handled in the improved training script.")
        else:
            print("\n✅ Output resolution matches input (no upsampling needed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_label_issues(dataset, num_samples=10):
    """Check for potential label issues"""
    print(f"\n{'='*80}")
    print("Checking Label Quality")
    print(f"{'='*80}")
    
    issues = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        label = sample['label']
        image = sample['image']
        
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # Check for unexpected values
        unique_labels = np.unique(label)
        if any(l > 6 or l < 0 for l in unique_labels):
            issues.append(f"Sample {i}: Invalid label values: {unique_labels}")
        
        # Check if predominantly one class
        counts = np.bincount(label.flatten(), minlength=7)
        if counts[1] > 0.98 * label.size:  # >98% is crop
            issues.append(f"Sample {i}: Almost all pixels are 'crop' class")
    
    if issues:
        print("\n⚠️ Potential issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print("✅ No obvious label issues detected")
    
    return issues


def visualize_sample(dataset, index=0, save_path=None):
    """Visualize a sample from the dataset"""
    print(f"\n{'='*80}")
    print(f"Visualizing Sample {index}")
    print(f"{'='*80}")
    
    sample = dataset[index]
    image = sample['image']
    label = sample['label']
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.numpy()
    else:
        image_np = image
    
    if isinstance(label, torch.Tensor):
        label_np = label.numpy()
    else:
        label_np = label
    
    # If image is normalized, denormalize it
    if image_np.min() < 0:
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406, 0.5])
        std = np.array([0.229, 0.224, 0.225, 0.25])
        
        image_np = image_np * std[:, None, None] + mean[:, None, None]
        image_np = np.clip(image_np, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    rgb = image_np[:3].transpose(1, 2, 0)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # NIR
    nir = image_np[3]
    axes[1].imshow(nir, cmap='gray')
    axes[1].set_title('NIR Channel')
    axes[1].axis('off')
    
    # Label
    im = axes[2].imshow(label_np, cmap='tab10', vmin=0, vmax=6)
    axes[2].set_title('Ground Truth Labels')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_ticks(range(7))
    cbar.set_ticklabels([
        'bg', 'crop', 'skip', 'water', 'weed', 'nutrient', 'waterway'
    ])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.savefig('/home/claude/sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to /home/claude/sample_visualization.png")
    
    plt.close()
    
    # Print stats
    unique, counts = np.unique(label_np, return_counts=True)
    print(f"\nLabel distribution in this sample:")
    class_names = ['background', 'crop', 'planter_skip', 'water', 
                   'weed_cluster', 'nutrient_deficiency', 'waterway']
    for cls, count in zip(unique, counts):
        pct = (count / label_np.size) * 100
        cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {cls}"
        print(f"  {cls_name}: {count:,} pixels ({pct:.2f}%)")


def main():
    """Run all diagnostics"""
    dataset_path = "dataset/finetuning_dataset/agriculture_vision_2019"
    
    print(f"\n{'='*80}")
    print("AGRICULTURE-VISION DATASET DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Dataset path: {dataset_path}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AgricultureVisionDataset(
        root_dir=dataset_path,
        split='train',
        transform=get_train_transforms()
    )
    
    val_dataset = AgricultureVisionDataset(
        root_dir=dataset_path,
        split='val',
        transform=get_val_transforms()
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 1. Analyze class distribution
    train_counts, train_weights = analyze_dataset_distribution(
        train_dataset, "Training Set", sample_size=200
    )
    
    # 2. Test model output shape
    test_model_output_shape(dataset_path)
    
    # 3. Check label quality
    check_label_issues(train_dataset, num_samples=20)
    
    # 4. Visualize sample
    visualize_sample(train_dataset, index=0)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("""
1. ✅ Use the IMPROVED training script (train_agriculture_vision_improved.py)
   - It handles output resolution mismatches automatically
   - Includes class weighting for imbalanced data
   - Better gradient clipping and stability
   
2. ✅ Use the IMPROVED config (agriculture_vision_v2.yaml)
   - Lower learning rate (5e-5 instead of 1e-4)
   - Class weights enabled
   - Better augmentation settings
   
3. 📊 Monitor per-class metrics carefully
   - The crop class will dominate overall metrics
   - Focus on F1 scores for anomaly classes (2-6)
   
4. 🎯 Expected initial performance:
   - Crop class: High F1 (>0.8)
   - Rare anomaly classes: Low F1 (<0.3 initially)
   - Overall mean F1: ~0.4-0.5 after a few epochs
   
5. ⏱️ Training time:
   - Expect 5-10 epochs before seeing meaningful anomaly detection
   - Early stopping patience should be at least 15-20 epochs
   
6. 🔍 If still having issues:
   - Try freezing encoder (freeze_encoder: true)
   - Reduce batch size if OOM errors occur
   - Increase learning rate slightly if loss doesn't decrease
    """)
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()