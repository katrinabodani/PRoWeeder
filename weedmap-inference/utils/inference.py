"""
Model Inference Utilities
Handles loading and running inference for all three models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import SegformerForImageClassification
from roweeder.models import RoWeederFlat

from .preprocessing import preprocess_image


class ModelInference:
    """Unified interface for all models"""
    
    def __init__(self, model_type, model_path, num_channels=3):
        """
        Initialize model
        
        Args:
            model_type: 'baseline', 'reproduced', or 'phase1'
            model_path: Path to model checkpoint or HF model name
            num_channels: Number of input channels (3 for RGB, 4 for RGB+NIR)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.num_channels = num_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        """Load the appropriate model based on type"""
        
        if self.model_type == 'baseline':
            # Load from HuggingFace
            print(f"Loading baseline model from HuggingFace: {self.model_path}")
            model = RoWeederFlat.from_pretrained(self.model_path)
            
        elif self.model_type == 'reproduced':
            # Load reproduced model checkpoint
            print(f"Loading reproduced model from: {self.model_path}")
            
            # Check if file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Reproduced model not found at {self.model_path}. "
                    "Please ensure training has completed and checkpoint is copied."
                )
            
            # Load checkpoint
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create base model architecture (3 channels, 3 classes - default)
            model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
            
            # Load trained weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
        elif self.model_type == 'phase1':
            # Load Phase 1 finetuned model
            # Architecture: 4-channel input (RGB+NIR), 7 output classes
            print(f"Loading Phase 1 finetuned model from: {self.model_path}")
            
            checkpoint_path = Path(self.model_path)
            
            # Check if file exists
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Phase 1 checkpoint not found at {self.model_path}. "
                    "Please ensure the checkpoint file exists at this exact path."
                )
            
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create base model from pretrained (same as training)
            print("Creating model architecture from pretrained base...")
            model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
            
            # Patch encoder for 4 channels (same as training)
            print("Patching encoder for 4-channel input (RGB+NIR)...")
            self._patch_encoder_for_4channels(model)
            
            # Replace output head for 7 classes (same as training)
            print("Replacing output head for 7 classes...")
            self._replace_output_head(model, num_classes=7)
            
            # Load finetuned weights
            print("Loading finetuned weights...")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print("✅ Phase 1 model loaded successfully")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _patch_encoder_for_4channels(self, model):
        """Patch the first conv layer to accept 4-channel input (matches training script)"""
        try:
            patch_embed = None
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'segformer'):
                segformer = model.encoder.segformer
                seg_enc = getattr(segformer, 'encoder', segformer)
                if hasattr(seg_enc, 'patch_embeddings'):
                    patch_embed = seg_enc.patch_embeddings[0]
            
            if patch_embed is None:
                raise AttributeError("Could not locate patch_embeddings")
            
            old_conv = patch_embed.proj
            if not isinstance(old_conv, nn.Conv2d):
                raise TypeError(f"Expected Conv2d, got {type(old_conv)}")
            
            # Create new 4-channel conv
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            )
            
            # Copy weights (checkpoint will override these anyway)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True).clone()
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            
            patch_embed.proj = new_conv
            
        except Exception as e:
            raise RuntimeError(f"Failed to patch encoder for 4 channels: {e}")
    
    def _replace_output_head(self, model, num_classes):
        """Replace output head with specified number of classes (matches training script)"""
        replaced = False
        
        # Try classifier attribute
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            if isinstance(classifier, nn.Conv2d):
                in_ch = classifier.in_channels
                model.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                replaced = True
            elif isinstance(classifier, (nn.Sequential, nn.ModuleList)):
                for i in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[i], nn.Conv2d):
                        in_ch = classifier[i].in_channels
                        classifier[i] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                        replaced = True
                        break
        
        # Try segmentation_head attribute
        if not replaced and hasattr(model, 'segmentation_head'):
            head = model.segmentation_head
            if isinstance(head, nn.Conv2d):
                in_ch = head.in_channels
                model.segmentation_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                replaced = True
            elif isinstance(head, (nn.Sequential, nn.ModuleList)):
                for i in range(len(head) - 1, -1, -1):
                    if isinstance(head[i], nn.Conv2d):
                        in_ch = head[i].in_channels
                        head[i] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                        replaced = True
                        break
        
        # Try decode_head attribute
        if not replaced and hasattr(model, 'decode_head'):
            head = model.decode_head
            if isinstance(head, nn.Conv2d):
                in_ch = head.in_channels
                model.decode_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                replaced = True
            elif isinstance(head, (nn.Sequential, nn.ModuleList)):
                for i in range(len(head) - 1, -1, -1):
                    if isinstance(head[i], nn.Conv2d):
                        in_ch = head[i].in_channels
                        head[i] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                        replaced = True
                        break
        
        if not replaced:
            raise RuntimeError(f"Could not find output head to replace with {num_classes} classes")
    
    def predict(self, image, map_to_3_classes=True):
        """
        Run inference on image
        
        Args:
            image: PIL Image
            map_to_3_classes: If True and model is Phase 1, map 7 classes to 3.
                             If False and model is Phase 1, keep all 7 classes.
            
        Returns:
            numpy array of shape (H, W) with class predictions
        """
        # Preprocess image
        input_tensor = preprocess_image(
            image,
            model_type=self.model_type,
            num_channels=self.num_channels,
            device=self.device
        )
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('out', list(outputs.values())[0]))
            else:
                logits = outputs
            
            # Get predictions
            if self.model_type == 'phase1':
                # Phase 1 outputs 7 classes
                preds = logits.argmax(dim=1).cpu().numpy()[0]
                
                # Map to 3 classes if requested
                if map_to_3_classes:
                    preds = self._map_phase1_to_weedmap(preds)
                # Otherwise keep all 7 classes for Agriculture-Vision display
            else:
                # Baseline and Reproduced output 3 classes directly
                preds = logits.argmax(dim=1).cpu().numpy()[0]
        
        return preds
    
    def _map_phase1_to_weedmap(self, phase1_preds):
        """
        Map Phase 1 predictions (7 classes) to WeedMap format (3 classes)
        
        Phase 1 classes (Agriculture-Vision):
            0: background
            1: crop
            2: planter_skip
            3: water
            4: weed_cluster
            5: nutrient_deficiency
            6: waterway
        
        WeedMap classes: 
            0: background
            1: crop
            2: weed
        """
        mapped = np.zeros_like(phase1_preds)
        
        # Map to WeedMap classes
        mapped[phase1_preds == 0] = 0  # background → background
        mapped[phase1_preds == 1] = 1  # crop → crop
        mapped[phase1_preds == 2] = 0  # planter_skip → background (empty spot)
        mapped[phase1_preds == 3] = 0  # water → background
        mapped[phase1_preds == 4] = 2  # weed_cluster → weed
        mapped[phase1_preds == 5] = 2  # nutrient_deficiency → weed (anomaly)
        mapped[phase1_preds == 6] = 0  # waterway → background
        
        return mapped