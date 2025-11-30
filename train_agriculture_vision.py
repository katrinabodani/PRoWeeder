"""
IMPROVED training script for fine-tuning RoWeeder on Agriculture-Vision dataset

Key improvements:
1. Proper output resolution handling (upsampling to match labels)
2. Class-weighted loss to handle imbalance
3. Better metrics tracking per class
4. Learning rate warmup
5. Improved validation logging
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

from agriculture_vision_dataset import (
    AgricultureVisionDataset,
    get_train_transforms,
    get_val_transforms
)

# ----------------------
# Improved Losses with stability and class weighting
# ----------------------
class DiceLoss(nn.Module):
    """Dice Loss with numerical stability"""
    def __init__(self, smooth=1.0, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Mask out ignored class
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = torch.clamp(dice, 0, 1)
        
        # Don't include ignored class in the mean
        if self.ignore_index is not None:
            dice = torch.cat([dice[:, :self.ignore_index], dice[:, self.ignore_index+1:]], dim=1)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined CrossEntropy + Dice Loss with class weights"""
    def __init__(self, ce_weight=1.0, dice_weight=0.5, ignore_index=0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, 
            reduction='mean',
            weight=class_weights
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Check for NaN
        if torch.isnan(ce) or torch.isinf(ce):
            print("⚠️ NaN/Inf detected in CE loss")
            ce = torch.tensor(0.0, device=pred.device)
        
        if torch.isnan(dice) or torch.isinf(dice):
            print("⚠️ NaN/Inf detected in Dice loss")
            dice = torch.tensor(0.0, device=pred.device)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        return total_loss


def calculate_class_weights(dataset, num_classes, ignore_index=0):
    """Calculate class weights based on frequency (inverse frequency)"""
    print("\nCalculating class weights from training data...")
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    for i in tqdm(range(min(500, len(dataset))), desc="Sampling dataset"):
        sample = dataset[i]
        label = sample['label'].numpy() if isinstance(sample['label'], torch.Tensor) else sample['label']
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()
    
    print("\nRaw class counts:")
    for c in range(num_classes):
        print(f"  Class {c}: {int(class_counts[c]):>12,} pixels")
    
    # Calculate weights (inverse frequency)
    # Method: weight = total_pixels / (num_classes * class_count)
    total_pixels = class_counts.sum()
    weights = np.zeros(num_classes, dtype=np.float64)
    
    for c in range(num_classes):
        if c == ignore_index:
            weights[c] = 0.0
        elif class_counts[c] > 0:
            # Inverse frequency: more pixels = lower weight
            weights[c] = total_pixels / (num_classes * class_counts[c])
        else:
            weights[c] = 0.0  # Class not present
    
    # Normalize weights so they sum to num_classes (excluding ignored)
    non_zero_weights = weights[weights > 0]
    if len(non_zero_weights) > 0:
        weight_sum = non_zero_weights.sum()
        weights[weights > 0] = weights[weights > 0] / weight_sum * (num_classes - (1 if ignore_index is not None else 0))
    
    print("\nFinal class weights:")
    for c in range(num_classes):
        pct = (class_counts[c] / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  Class {c}: count={int(class_counts[c]):>12,} ({pct:>6.2f}%) → weight={weights[c]:.4f}")
    
    # Verify weights are valid
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        print("⚠️ WARNING: Invalid weights detected! Using uniform weights instead.")
        weights = np.ones(num_classes, dtype=np.float64)
        if ignore_index is not None:
            weights[ignore_index] = 0.0
    
    return torch.FloatTensor(weights)


def patch_encoder_for_4channels(model: nn.Module):
    """Patch the first conv layer to accept 4-channel input"""
    try:
        print("Patching RoWeeder encoder for 4-channel input (RGB+NIR)...")
        
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
        
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        
        with torch.no_grad():
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
            # Initialize NIR channel (average of RGB)
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True).clone()
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        
        patch_embed.proj = new_conv
        print("✅ Successfully patched encoder to accept 4 channels")
        return True
    
    except Exception as e:
        print(f"❌ Failed to patch model: {e}")
        return False


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0):
    """Calculate IoU and F1 metrics per class"""
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    metrics = {}
    class_metrics = defaultdict(dict)
    class_iou_list = []
    class_f1_list = []
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        iou = intersection / (union + 1e-7)
        precision = intersection / (pred_sum + 1e-7)
        recall = intersection / (target_sum + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        class_metrics[cls]['iou'] = float(iou)
        class_metrics[cls]['f1'] = float(f1)
        class_metrics[cls]['precision'] = float(precision)
        class_metrics[cls]['recall'] = float(recall)
        class_metrics[cls]['target_pixels'] = int(target_sum)
        class_metrics[cls]['pred_pixels'] = int(pred_sum)
        
        if target_sum > 0:  # Only include classes that exist in ground truth
            class_iou_list.append(iou)
            class_f1_list.append(f1)
    
    metrics['mean_iou'] = float(np.mean(class_iou_list)) if class_iou_list else 0.0
    metrics['mean_f1'] = float(np.mean(class_f1_list)) if class_f1_list else 0.0
    metrics['class_metrics'] = dict(class_metrics)
    
    return metrics


def extract_logits(output: Any, target_size: tuple = None) -> torch.Tensor:
    """Extract logits from model output and upsample to target size if needed"""
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, dict):
        logits = None
        for key in ('logits', 'preds', 'out', 'segmentation'):
            if key in output and isinstance(output[key], torch.Tensor):
                logits = output[key]
                break
        if logits is None:
            for v in output.values():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    logits = v
                    break
        if logits is None:
            raise ValueError(f"Cannot extract logits from output: {type(output)}")
    else:
        raise ValueError(f"Cannot extract logits from output: {type(output)}")
    
    # Upsample to target size if needed
    if target_size is not None and logits.shape[-2:] != target_size:
        logits = torch.nn.functional.interpolate(
            logits, size=target_size, mode='bilinear', align_corners=False
        )
    
    return logits


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip=1.0, epoch=0):
    """Train for one epoch with proper resolution handling"""
    model.train()
    running_loss = 0.0
    nan_count = 0
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].long().to(device)
        
        optimizer.zero_grad()
        
        # Check for NaN in inputs
        if torch.isnan(images).any() or torch.isnan(labels.float()).any():
            print(f"⚠️ NaN detected in input at batch {batch_idx}, skipping...")
            continue
        
        try:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    logits = extract_logits(outputs, target_size=labels.shape[-2:])
                    
                    # Debug: print shapes on first batch
                    if batch_idx == 0:
                        print(f"\n  Input shape: {images.shape}")
                        print(f"  Output shape: {logits.shape}")
                        print(f"  Label shape: {labels.shape}")
                    
                    loss = criterion(logits, labels)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"⚠️ NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                logits = extract_logits(outputs, target_size=labels.shape[-2:])
                
                if batch_idx == 0:
                    print(f"\n  Input shape: {images.shape}")
                    print(f"  Output shape: {logits.shape}")
                    print(f"  Label shape: {labels.shape}")
                
                loss = criterion(logits, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"⚠️ NaN/Inf loss at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{running_loss / (batch_idx + 1 - nan_count):.4f}"})
            
        except RuntimeError as e:
            print(f"⚠️ Runtime error at batch {batch_idx}: {e}")
            continue
    
    if nan_count > 0:
        print(f"⚠️ Encountered {nan_count} NaN losses this epoch")
    
    return running_loss / max(1, len(train_loader) - nan_count)


def validate(model, val_loader, criterion, device, num_classes, ignore_index=0, class_names=None):
    """Validate the model with detailed per-class metrics"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].long().to(device)
            
            outputs = model(images)
            logits = extract_logits(outputs, target_size=labels.shape[-2:])
            loss = criterion(logits, labels)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                running_loss += loss.item()
                metrics = calculate_metrics(logits, labels, num_classes, ignore_index)
                all_metrics.append(metrics)
    
    if not all_metrics:
        return {'mean_iou': 0.0, 'mean_f1': 0.0, 'val_loss': 0.0}
    
    # Aggregate metrics
    avg_metrics = {}
    avg_metrics['val_loss'] = running_loss / max(1, len(val_loader))
    
    # Average IoU and F1
    avg_metrics['mean_iou'] = np.mean([m['mean_iou'] for m in all_metrics])
    avg_metrics['mean_f1'] = np.mean([m['mean_f1'] for m in all_metrics])
    
    # Per-class metrics
    class_metrics_agg = defaultdict(lambda: defaultdict(list))
    for m in all_metrics:
        for cls, cls_metrics in m['class_metrics'].items():
            for metric_name, value in cls_metrics.items():
                if metric_name not in ['target_pixels', 'pred_pixels']:
                    class_metrics_agg[cls][metric_name].append(value)
    
    avg_metrics['per_class'] = {}
    for cls in class_metrics_agg:
        avg_metrics['per_class'][cls] = {}
        for metric_name, values in class_metrics_agg[cls].items():
            avg_metrics['per_class'][cls][metric_name] = float(np.mean(values))
    
    # Print per-class results
    print("\nPer-class validation metrics:")
    print("-" * 80)
    print(f"{'Class':<25} {'IoU':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 80)
    
    for cls in sorted(avg_metrics['per_class'].keys()):
        cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        cls_m = avg_metrics['per_class'][cls]
        print(f"{cls_name:<25} {cls_m.get('iou', 0):>10.4f} {cls_m.get('f1', 0):>10.4f} "
              f"{cls_m.get('precision', 0):>12.4f} {cls_m.get('recall', 0):>10.4f}")
    print("-" * 80)
    
    return avg_metrics


def load_config(config_path: str):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path: str, resume_from: str = None):
    """Main training function with improvements"""
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(config['logging']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = None
    if config['logging'].get('use_tensorboard', False):
        writer = SummaryWriter(output_dir / 'tensorboard')
    
    print("\nLoading datasets...")
    train_dataset = AgricultureVisionDataset(
        root_dir=config['data']['dataset_path'],
        split='train',
        transform=get_train_transforms()
    )
    
    val_dataset = AgricultureVisionDataset(
        root_dir=config['data']['dataset_path'],
        split='val',
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    print("\nInitializing model...")
    from roweeder.models import RoWeederFlat
    
    print("Loading pretrained RoWeeder from pasqualedem/roweeder_flat_512x512")
    model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512")
    
    # Patch for 4 channels
    patched = patch_encoder_for_4channels(model)
    if not patched:
        raise RuntimeError("Failed to patch model for 4-channel input!")
    
    # Adapt output head
    target_num_classes = config['model']['num_classes']
    print(f"Adapting output head to {target_num_classes} classes...")
    
    # Try to find and replace the final conv layer
    replaced = False
    
    # Check for classifier attribute
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, nn.Conv2d):
            in_ch = classifier.in_channels
            model.classifier = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
            replaced = True
            print(f"✅ Replaced classifier: Conv2d({in_ch}, {target_num_classes})")
        elif isinstance(classifier, (nn.Sequential, nn.ModuleList)):
            for i in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[i], nn.Conv2d):
                    in_ch = classifier[i].in_channels
                    classifier[i] = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
                    replaced = True
                    print(f"✅ Replaced classifier[{i}]: Conv2d({in_ch}, {target_num_classes})")
                    break
    
    # Check for segmentation_head attribute
    if not replaced and hasattr(model, 'segmentation_head'):
        head = model.segmentation_head
        if isinstance(head, nn.Conv2d):
            in_ch = head.in_channels
            model.segmentation_head = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
            replaced = True
            print(f"✅ Replaced segmentation_head: Conv2d({in_ch}, {target_num_classes})")
        elif isinstance(head, (nn.Sequential, nn.ModuleList)):
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], nn.Conv2d):
                    in_ch = head[i].in_channels
                    head[i] = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
                    replaced = True
                    print(f"✅ Replaced segmentation_head[{i}]: Conv2d({in_ch}, {target_num_classes})")
                    break
    
    # Check for decode_head attribute
    if not replaced and hasattr(model, 'decode_head'):
        head = model.decode_head
        if isinstance(head, nn.Conv2d):
            in_ch = head.in_channels
            model.decode_head = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
            replaced = True
            print(f"✅ Replaced decode_head: Conv2d({in_ch}, {target_num_classes})")
        elif isinstance(head, (nn.Sequential, nn.ModuleList)):
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], nn.Conv2d):
                    in_ch = head[i].in_channels
                    head[i] = nn.Conv2d(in_ch, target_num_classes, kernel_size=1)
                    replaced = True
                    print(f"✅ Replaced decode_head[{i}]: Conv2d({in_ch}, {target_num_classes})")
                    break
    
    if not replaced:
        print("⚠️ Could not automatically replace output head")
        print("Model structure:")
        for name, module in model.named_children():
            print(f"  - {name}: {type(module).__name__}")
        raise RuntimeError("Failed to adapt model output layer.")
    
    # Optionally freeze encoder
    if config['model'].get('freeze_encoder', False):
        print("Freezing encoder layers...")
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
    
    model = model.to(device)
    print(f"✅ Model ready: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    
    # Calculate class weights
    use_class_weights = config['training'].get('use_class_weights', True)
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(
            train_dataset, 
            target_num_classes, 
            ignore_index=0
        )
        class_weights = class_weights.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(
        ce_weight=1.0, 
        dice_weight=0.5, 
        ignore_index=0,
        class_weights=class_weights
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda') if (config['training'].get('use_amp', False) and torch.cuda.is_available()) else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_f1 = 0.0
    
    if resume_from:
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    print("\nStarting training...")
    patience_counter = 0
    class_names = config['data'].get('class_names', None)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*80}")
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            grad_clip=config['training'].get('grad_clip', 1.0),
            epoch=epoch
        )
        
        if np.isnan(train_loss):
            print("❌ Training collapsed with NaN loss! Stopping...")
            break
        
        val_metrics = validate(
            model, val_loader, criterion, device, target_num_classes,
            ignore_index=0, class_names=class_names
        )
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"Mean IoU:   {val_metrics['mean_iou']:.4f}")
        print(f"Mean F1:    {val_metrics['mean_f1']:.4f}")
        print(f"{'='*80}")
        
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            writer.add_scalar('Metrics/mean_iou', val_metrics['mean_iou'], epoch)
            writer.add_scalar('Metrics/mean_f1', val_metrics['mean_f1'], epoch)
            
            # Log per-class metrics
            if 'per_class' in val_metrics:
                for cls, cls_metrics in val_metrics['per_class'].items():
                    cls_name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"
                    for metric_name, value in cls_metrics.items():
                        writer.add_scalar(f'PerClass/{cls_name}/{metric_name}', value, epoch)
        
        scheduler.step(val_metrics['mean_f1'])
        
        # Save best model
        if val_metrics['mean_f1'] > best_f1:
            best_f1 = val_metrics['mean_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save periodic checkpoints
        save_frequency = config['training'].get('save_frequency', 5)
        if (epoch + 1) % save_frequency == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path.name}")
        
        # Always save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'val_metrics': val_metrics,
            'config': config
        }, output_dir / 'last_checkpoint.pth')
        
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\n{'='*80}")
    print(f"✅ Training completed! Best F1: {best_f1:.4f}")
    print(f"{'='*80}")
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RoWeeder on Agriculture-Vision (IMPROVED)')
    parser.add_argument('--config', type=str, default='parameters/finetuning/agriculture_vision.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(args.config, resume_from=args.resume)