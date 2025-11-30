"""
Image Preprocessing Utilities
Handle different preprocessing for each model type
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image, model_type, num_channels=3, device='cpu'):
    """
    Preprocess image for specific model
    
    Args:
        image: PIL Image
        model_type: 'baseline', 'reproduced', or 'phase1'
        num_channels: Number of channels (3 for RGB, 4 for RGB+NIR)
        device: torch device
        
    Returns:
        torch.Tensor ready for model input
    """
    # Resize to 512×512
    image = image.resize((512, 512), Image.BILINEAR)
    
    if model_type in ['baseline', 'reproduced']:
        # RGB only (3 channels)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
    elif model_type == 'phase1':
        # RGB + NIR (4 channels)
        # For Phase 1, we need 4 channels
        # If input image is RGB only, duplicate one channel as NIR
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Create 4-channel array
        # Use Red channel as NIR approximation (common in agriculture)
        nir_channel = img_array[:, :, 0:1]  # Use R channel
        img_array_4ch = np.concatenate([img_array, nir_channel], axis=-1)
        
        # Apply normalization (extend to 4 channels)
        # Use ImageNet stats for RGB, same std for NIR
        mean = np.array([0.485, 0.456, 0.406, 0.485], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225, 0.229], dtype=np.float32)
        img_array_4ch = (img_array_4ch - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(img_array_4ch).permute(2, 0, 1).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return tensor.to(device)


def preprocess_for_comparison(image, device='cpu'):
    """
    Preprocess image for all models (returns dict)
    
    Args:
        image: PIL Image
        device: torch device
        
    Returns:
        dict of tensors for each model type
    """
    return {
        'baseline': preprocess_image(image, 'baseline', 3, device),
        'reproduced': preprocess_image(image, 'reproduced', 3, device),
        'phase1': preprocess_image(image, 'phase1', 4, device),
    }


def load_nir_channel(nir_path):
    """
    Load NIR channel from separate file (if available)
    
    Args:
        nir_path: Path to NIR image file
        
    Returns:
        numpy array of NIR channel
    """
    nir_image = Image.open(nir_path).convert('L')  # Grayscale
    nir_image = nir_image.resize((512, 512), Image.BILINEAR)
    nir_array = np.array(nir_image).astype(np.float32) / 255.0
    return nir_array


def create_rgb_nir_composite(rgb_image, nir_image):
    """
    Create 4-channel composite from RGB and NIR images
    
    Args:
        rgb_image: PIL Image (RGB)
        nir_image: PIL Image (grayscale NIR)
        
    Returns:
        numpy array of shape (H, W, 4)
    """
    # Resize both to 512×512
    rgb_image = rgb_image.resize((512, 512), Image.BILINEAR)
    nir_image = nir_image.resize((512, 512), Image.BILINEAR)
    
    # Convert to arrays
    rgb_array = np.array(rgb_image.convert('RGB')).astype(np.float32) / 255.0
    nir_array = np.array(nir_image.convert('L')).astype(np.float32) / 255.0
    
    # Add NIR as 4th channel
    nir_array = np.expand_dims(nir_array, axis=-1)
    composite = np.concatenate([rgb_array, nir_array], axis=-1)
    
    return composite