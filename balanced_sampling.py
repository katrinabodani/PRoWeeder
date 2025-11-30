"""
Balanced Sampling for Agriculture-Vision Dataset

This creates a balanced sampler that:
1. Identifies images with rare classes (weed, planter_skip, nutrient_def)
2. Oversamples these images during training
3. Ensures model sees rare classes more frequently
"""
import torch
import numpy as np
from torch.utils.data import Sampler
from collections import Counter
from tqdm import tqdm


class BalancedClassSampler(Sampler):
    """
    Sampler that oversamples images containing rare classes
    
    Strategy:
    - Images with rare classes (weed, planter_skip, nutrient_def): sampled more
    - Images with only common classes (crop, water): sampled less
    - Creates balanced distribution across epochs
    """
    
    def __init__(self, dataset, rare_class_ids=[2, 4, 5], oversample_factor=5):
        """
        Args:
            dataset: Agriculture-Vision dataset
            rare_class_ids: List of rare class IDs (default: planter_skip, weed, nutrient_def)
            oversample_factor: How many times to oversample rare class images
        """
        self.dataset = dataset
        self.rare_class_ids = rare_class_ids
        self.oversample_factor = oversample_factor
        
        print("\nAnalyzing dataset for balanced sampling...")
        self._analyze_dataset()
        self._create_sampling_weights()
    
    def _analyze_dataset(self):
        """Analyze which images contain rare classes"""
        self.rare_class_images = {cls: [] for cls in self.rare_class_ids}
        self.common_images = []
        
        print("Scanning images for rare classes...")
        for idx in tqdm(range(len(self.dataset)), desc="Analyzing"):
            sample = self.dataset[idx]
            label = sample['label']
            
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            
            unique_classes = np.unique(label)
            
            # Check if image contains any rare class
            has_rare = False
            for rare_cls in self.rare_class_ids:
                if rare_cls in unique_classes:
                    self.rare_class_images[rare_cls].append(idx)
                    has_rare = True
            
            if not has_rare:
                self.common_images.append(idx)
        
        print("\nDataset composition:")
        print(f"  Common images (crop/water only): {len(self.common_images)}")
        for cls, indices in self.rare_class_images.items():
            class_names = ['bg', 'crop', 'planter_skip', 'water', 'weed', 'nutrient_def', 'waterway']
            print(f"  Images with {class_names[cls]}: {len(indices)}")
    
    def _create_sampling_weights(self):
        """Create sampling weights for each image"""
        self.weights = np.ones(len(self.dataset))
        
        # Increase weight for images with rare classes
        for rare_cls, indices in self.rare_class_images.items():
            for idx in indices:
                self.weights[idx] *= self.oversample_factor
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        print(f"\nSampling strategy:")
        print(f"  Rare class oversample factor: {self.oversample_factor}x")
        print(f"  Rare images will be seen ~{self.oversample_factor}x more often")
    
    def __iter__(self):
        """Generate sampling indices"""
        # Sample with replacement based on weights
        indices = np.random.choice(
            len(self.dataset),
            size=len(self.dataset),
            replace=True,
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss - automatically focuses on hard examples
    
    Better than class weights for extreme imbalance because:
    - Focuses on hard-to-classify examples
    - Reduces importance of easy examples
    - Works well with rare classes
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=0):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore in loss calculation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (B, C, H, W)
            targets: ground truth (B, H, W)
        """
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        
        # Get probabilities
        probs = torch.exp(log_probs)
        
        # Gather probabilities for true class
        targets_one_hot = torch.nn.functional.one_hot(
            targets.long(), 
            num_classes=inputs.shape[1]
        ).permute(0, 3, 1, 2).float()
        
        # Get probability of correct class
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy loss
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Get weights for each pixel
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Mask out ignored index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            focal_loss = focal_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-7)
        
        return focal_loss.mean()


# Usage example in training script
def get_balanced_dataloader(dataset, batch_size=4, num_workers=0):
    """
    Create a dataloader with balanced sampling
    
    Usage:
        train_loader = get_balanced_dataloader(
            train_dataset, 
            batch_size=4, 
            num_workers=0
        )
    """
    sampler = BalancedClassSampler(
        dataset,
        rare_class_ids=[2, 4, 5],  # planter_skip, weed, nutrient_def
        oversample_factor=5  # See rare class images 5x more often
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use custom sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    # Test the sampler
    from agriculture_vision_dataset import AgricultureVisionDataset, get_train_transforms
    
    dataset_path = "dataset/finetuning_dataset/agriculture_vision_2019"
    
    print("Loading dataset...")
    train_dataset = AgricultureVisionDataset(
        root_dir=dataset_path,
        split='train',
        transform=get_train_transforms()
    )
    
    print("\nCreating balanced sampler...")
    sampler = BalancedClassSampler(
        train_dataset,
        rare_class_ids=[2, 4, 5],
        oversample_factor=5
    )
    
    print("\n" + "="*80)
    print("Testing balanced sampling...")
    print("="*80)
    
    # Sample 100 images and see distribution
    test_indices = [next(iter(sampler)) for _ in range(100)]
    
    rare_count = 0
    for idx in test_indices:
        sample = train_dataset[idx]
        label = sample['label'].numpy()
        unique = np.unique(label)
        
        if any(cls in unique for cls in [2, 4, 5]):
            rare_count += 1
    
    print(f"\nIn 100 sampled images:")
    print(f"  Images with rare classes: {rare_count}/100")
    print(f"  Expected without balancing: ~10-15/100")
    print(f"  With balancing: ~{rare_count}/100")
    
    if rare_count > 40:
        print("\n✅ Balancing is working! Rare classes oversampled.")
    else:
        print("\n⚠️ Balancing might need tuning.")