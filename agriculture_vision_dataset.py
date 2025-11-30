import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AgricultureVisionDataset(Dataset):
    """
    Dataset for Agriculture-Vision 2019
    Loads RGB + NIR images and combines multiple anomaly masks into single label
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to agriculture_vision_2019 folder
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load split counts
        split_file = os.path.join(root_dir, 'split_stats.json')
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Get all image IDs from RGB folder
        rgb_dir = os.path.join(root_dir, 'field_images', 'rgb')
        all_image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        all_image_ids = [f.replace('.jpg', '') for f in all_image_files]
        
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
        if split == 'train':
            self.image_ids = all_image_ids[:train_count]
        elif split == 'val':
            self.image_ids = all_image_ids[train_count:train_count + val_count]
        else:  # test
            self.image_ids = all_image_ids[train_count + val_count:]
        
        # Anomaly types (only the 5 active ones)
        self.anomaly_types = [
            'planter_skip',
            'water', 
            'weed_cluster',
            'nutrient_deficiency',
            'waterway'
        ]
        
        # Class mapping
        # 0: background (invalid pixels)
        # 1: crop (healthy field area)
        # 2: planter_skip
        # 3: water
        # 4: weed_cluster
        # 5: nutrient_deficiency
        # 6: waterway
        self.num_classes = 7
        
        print(f"✅ Loaded {split} split: {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load RGB image
        rgb_path = os.path.join(self.root_dir, 'field_images', 'rgb', f'{image_id}.jpg')
        rgb_img = np.array(Image.open(rgb_path))
        
        # Load NIR image
        nir_path = os.path.join(self.root_dir, 'field_images', 'nir', f'{image_id}.jpg')
        nir_img = np.array(Image.open(nir_path))
        
        # Combine RGB + NIR (4 channels)
        if len(nir_img.shape) == 3:
            nir_img = nir_img[:, :, 0]  # Take first channel if NIR is 3-channel
        
        # Ensure NIR is same size as RGB
        if nir_img.shape[:2] != rgb_img.shape[:2]:
            nir_img = np.array(Image.fromarray(nir_img).resize((rgb_img.shape[1], rgb_img.shape[0])))
        
        image = np.dstack([rgb_img, nir_img[:, :, np.newaxis] if len(nir_img.shape) == 2 else nir_img])  # (H, W, 4)
        
        # Load valid pixel mask
        mask_path = os.path.join(self.root_dir, 'field_masks', f'{image_id}.png')
        if os.path.exists(mask_path):
            valid_mask = np.array(Image.open(mask_path).convert('L'))
            valid_mask = (valid_mask > 0).astype(np.uint8)
        else:
            # If no mask, assume all pixels are valid
            valid_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Initialize label mask (all pixels start as crop=1)
        label = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Set invalid pixels to background (0)
        label[valid_mask == 0] = 0
        
        # Load anomaly masks and assign class IDs
        for idx, anomaly_type in enumerate(self.anomaly_types):
            anomaly_path = os.path.join(
                self.root_dir, 'field_labels', anomaly_type, f'{image_id}.png'
            )
            
            if os.path.exists(anomaly_path):
                anomaly_mask = np.array(Image.open(anomaly_path).convert('L'))
                anomaly_mask = (anomaly_mask > 0).astype(np.uint8)
                
                # Assign class ID (2-6 for anomalies)
                label[anomaly_mask > 0] = idx + 2
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        else:
            # If no transform, still convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label = torch.from_numpy(label).long()
        
        return {
            'image': image,
            'label': label,
            'image_id': image_id
        }


def get_train_transforms():
    """Training augmentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.1, 0.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],  # RGB + NIR
            std=[0.229, 0.224, 0.225, 0.25]
        ),
        ToTensorV2()
    ])


def get_val_transforms():
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],  # RGB + NIR
            std=[0.229, 0.224, 0.225, 0.25]
        ),
        ToTensorV2()
    ])


# Test the dataset
if __name__ == "__main__":
    dataset_path = "dataset/finetuning_dataset/agriculture_vision_2019"
    
    print("=" * 60)
    print("Testing Agriculture-Vision Dataset Loader")
    print("=" * 60)
    
    train_dataset = AgricultureVisionDataset(dataset_path, split='train', transform=get_train_transforms())
    val_dataset = AgricultureVisionDataset(dataset_path, split='val', transform=get_val_transforms())
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    # Test loading one sample
    print(f"\n{'='*60}")
    print("Testing sample loading...")
    print(f"{'='*60}")
    sample = train_dataset[0]
    print(f"✅ Image shape: {sample['image'].shape}")
    print(f"✅ Label shape: {sample['label'].shape}")
    print(f"✅ Unique labels: {torch.unique(sample['label']).tolist()}")
    print(f"✅ Image ID: {sample['image_id']}")
    
    # Test a few more samples
    print(f"\n{'='*60}")
    print("Testing multiple samples...")
    print(f"{'='*60}")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        unique_labels = torch.unique(sample['label']).tolist()
        print(f"Sample {i}: ID={sample['image_id']}, Labels={unique_labels}")
    
    print(f"\n{'='*60}")
    print("✅ Dataset loader working correctly!")
    print(f"{'='*60}")