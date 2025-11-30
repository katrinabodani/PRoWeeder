"""
Visualization Utilities
Create overlays, statistics plots, and comparisons
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO


def create_overlay(image, segmentation, class_colors, alpha=0.6):
    """
    Create segmentation overlay on original image
    
    Args:
        image: PIL Image (original)
        segmentation: numpy array (H, W) with class predictions
        class_colors: dict mapping class_id -> [R, G, B]
        alpha: overlay opacity (0-1)
        
    Returns:
        PIL Image with overlay
    """
    # Convert image to numpy
    img_array = np.array(image.convert('RGB'))
    
    # Resize segmentation to match image if needed
    if segmentation.shape != img_array.shape[:2]:
        seg_pil = Image.fromarray(segmentation.astype(np.uint8))
        seg_pil = seg_pil.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        segmentation = np.array(seg_pil)
    
    # Create colored segmentation mask
    h, w = segmentation.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        colored_mask[segmentation == class_id] = color
    
    # Blend with original image
    blended = (alpha * colored_mask + (1 - alpha) * img_array).astype(np.uint8)
    
    # Convert back to PIL
    overlay_img = Image.fromarray(blended)
    
    return overlay_img


def create_stats_plot(segmentation, class_names):
    """
    Create bar plot showing class distribution
    
    Args:
        segmentation: numpy array (H, W) with class predictions
        class_names: list of class names
        
    Returns:
        matplotlib figure
    """
    # Calculate class percentages
    total_pixels = segmentation.size
    class_counts = []
    class_percentages = []
    
    for class_id in range(len(class_names)):
        count = (segmentation == class_id).sum()
        percentage = (count / total_pixels) * 100
        class_counts.append(count)
        class_percentages.append(percentage)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Colors for bars
    colors = ['black', 'green', 'red']
    
    # Create bar plot
    bars = ax.bar(class_names, class_percentages, color=colors, alpha=0.7, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, class_percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Image', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(class_percentages) * 1.2)  # Add headroom for labels
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend with pixel counts
    legend_labels = [
        f'{name}: {count:,} pixels ({pct:.1f}%)'
        for name, count, pct in zip(class_names, class_counts, class_percentages)
    ]
    ax.legend(bars, legend_labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def create_comparison(original_image, results):
    """
    Create side-by-side comparison of multiple model results
    
    Args:
        original_image: PIL Image
        results: list of (model_name, overlay_image) tuples
        
    Returns:
        PIL Image with comparison grid
    """
    num_models = len(results)
    
    # Add original image to results
    all_images = [("Original", original_image)] + results
    
    # Calculate grid dimensions
    num_cols = min(3, len(all_images))  # Max 3 columns
    num_rows = (len(all_images) + num_cols - 1) // num_cols
    
    # Get image dimensions
    img_width, img_height = original_image.size
    
    # Create canvas
    canvas_width = num_cols * img_width
    canvas_height = num_rows * (img_height + 40)  # +40 for label
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    
    # Font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(canvas)
    
    # Place images on canvas
    for idx, (name, img) in enumerate(all_images):
        row = idx // num_cols
        col = idx % num_cols
        
        x = col * img_width
        y = row * (img_height + 40)
        
        # Paste image
        canvas.paste(img.resize((img_width, img_height)), (x, y + 40))
        
        # Draw label
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (img_width - text_width) // 2
        draw.text((text_x, y + 10), name, fill='black', font=font)
    
    return canvas


def create_legend():
    """
    Create a legend image for class colors
    
    Returns:
        PIL Image with color legend
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    
    # Create legend patches
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='green', label='Crop'),
        mpatches.Patch(color='red', label='Weed'),
    ]
    
    ax.legend(
        handles=patches,
        loc='center',
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    legend_img = Image.open(buf)
    plt.close(fig)
    
    return legend_img