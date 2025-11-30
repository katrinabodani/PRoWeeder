"""
WeedMap AI - Precision Agriculture Weed Detection
Semantic segmentation for crop and weed detection using deep learning

Models:
- Baseline RoWeeder (HF pretrained)
- Reproduced RoWeeder (Trained on WeedMap pseudo-labels)
- Phase 1 Finetuned (Trained on Agriculture-Vision)
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path

from utils.inference import ModelInference
from utils.visualization import create_overlay, create_stats_plot, create_comparison
from utils.preprocessing import preprocess_image

# Model configurations
MODELS = {
    "Baseline RoWeeder (HuggingFace)": {
        "type": "baseline",
        "path": "pasqualedem/roweeder_flat_512x512",
        "description": "Pretrained RoWeeder from HuggingFace",
        "channels": 3,
    },
    "Reproduced RoWeeder (Our Training)": {
        "type": "reproduced",
        "path": "models/reproduced_roweeder/best_checkpoint.pth",
        "description": "RoWeeder trained from scratch on WeedMap",
        "channels": 3,
    },
    "Phase 1 Finetuned (Agriculture-Vision)": {
        "type": "phase1",
        "path": "models/phase1_finetuned/best_model.pth",
        "description": "Finetuned on Agriculture-Vision dataset",
        "channels": 4,
    },
}

# Class names and colors for different datasets
CLASS_NAMES_WEEDMAP = ["Background", "Crop", "Weed"]
CLASS_COLORS_WEEDMAP = {
    0: [0, 0, 0],        # Background - Black
    1: [0, 255, 0],      # Crop - Green
    2: [255, 0, 0],      # Weed - Red
}

CLASS_NAMES_AGVISION = [
    "background",
    "crop",
    "planter_skip",
    "water",
    "weed_cluster",
    "nutrient_deficiency",
    "waterway"
]
CLASS_COLORS_AGVISION = {
    0: [0, 0, 0],         # Background - Black
    1: [0, 255, 0],       # Crop - Green
    2: [255, 165, 0],     # Planter Skip - Orange
    3: [0, 0, 255],       # Water - Blue
    4: [255, 0, 0],       # Weed Cluster - Red
    5: [255, 255, 0],     # Nutrient Deficiency - Yellow
    6: [0, 191, 255],     # Waterway - Light Blue
}

# Global model cache
model_cache = {}


def load_model(model_name):
    """Load model with caching"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    config = MODELS[model_name]
    model = ModelInference(
        model_type=config["type"],
        model_path=config["path"],
        num_channels=config["channels"]
    )
    model_cache[model_name] = model
    return model


def predict(image, model_name, dataset_type, overlay_opacity, show_stats):
    """
    Run inference on uploaded image
    
    Args:
        image: PIL Image or numpy array
        model_name: Name of model to use
        dataset_type: "WeedMap (3 classes)" or "Agriculture-Vision (7 classes)" or index
        overlay_opacity: Opacity of segmentation overlay (0-100)
        show_stats: Whether to show statistics plot
    
    Returns:
        Tuple of (overlay_image, stats_plot or None)
    """
    if image is None:
        return None, None
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Handle dataset_type - Gradio sometimes passes index instead of value
    if isinstance(dataset_type, int):
        options = ["WeedMap (3 classes)", "Agriculture-Vision (7 classes)"]
        dataset_type = options[dataset_type] if 0 <= dataset_type < len(options) else options[0]
    elif dataset_type is None:
        dataset_type = "WeedMap (3 classes)"  # Default
    
    # CRITICAL: Only Phase 1 on Agriculture-Vision shows 7 classes!
    # Baseline and Reproduced ALWAYS show 3 classes (not trained on 7)
    is_phase1 = "Phase 1" in model_name
    is_agvision = "Agriculture-Vision" in str(dataset_type)
    use_7_classes = is_phase1 and is_agvision
    
    # Select appropriate class names and colors
    if use_7_classes:
        class_names = CLASS_NAMES_AGVISION
        class_colors = CLASS_COLORS_AGVISION
    else:
        class_names = CLASS_NAMES_WEEDMAP
        class_colors = CLASS_COLORS_WEEDMAP
    
    # Load model
    model = load_model(model_name)
    
    # Run inference
    # map_to_3_classes: True for all except Phase 1 on Ag-Vision
    segmentation = model.predict(image, map_to_3_classes=(not use_7_classes))
    
    # Create overlay
    alpha = overlay_opacity / 100.0
    overlay_img = create_overlay(image, segmentation, class_colors, alpha=alpha)
    
    # Create stats plot if requested
    stats_plot = None
    if show_stats:
        stats_plot = create_stats_plot(segmentation, class_names)
    
    return overlay_img, stats_plot


def compare_models(image, models_to_compare, dataset_type, overlay_opacity):
    """
    Compare multiple models side-by-side
    
    Args:
        image: Input image
        models_to_compare: List of model names
        dataset_type: "WeedMap (3 classes)" or "Agriculture-Vision (7 classes)" or index
        overlay_opacity: Opacity for all overlays
    
    Returns:
        Comparison image with all models
    """
    if image is None or not models_to_compare:
        return None
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Handle dataset_type - Gradio sometimes passes index instead of value
    if isinstance(dataset_type, int):
        options = ["WeedMap (3 classes)", "Agriculture-Vision (7 classes)"]
        dataset_type = options[dataset_type] if 0 <= dataset_type < len(options) else options[0]
    elif dataset_type is None:
        dataset_type = "WeedMap (3 classes)"  # Default
    
    is_agvision = "Agriculture-Vision" in str(dataset_type)
    
    results = []
    
    # Run inference with each model
    for model_name in models_to_compare:
        # CRITICAL: Only Phase 1 on Agriculture-Vision shows 7 classes!
        is_phase1 = "Phase 1" in model_name
        use_7_classes = is_phase1 and is_agvision
        
        # Select appropriate class colors for THIS model
        if use_7_classes:
            class_colors = CLASS_COLORS_AGVISION
        else:
            class_colors = CLASS_COLORS_WEEDMAP
        
        model = load_model(model_name)
        segmentation = model.predict(image, map_to_3_classes=(not use_7_classes))
        alpha = overlay_opacity / 100.0
        overlay_img = create_overlay(image, segmentation, class_colors, alpha=alpha)
        results.append((model_name, overlay_img))
    
    # Create comparison
    comparison_img = create_comparison(image, results)
    
    return comparison_img


# Custom CSS for better styling
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}

#description {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 20px;
}

.model-info {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

footer {
    text-align: center;
    margin-top: 30px;
}
"""
# Create Gradio interface
with gr.Blocks(title="PRoWeeder") as demo:
    
    # Header
    gr.Markdown(
        """
        <div id="title">
        <h1> PRoWeeder Precision Agriculture</h1>
        <h3>Semantic Segmentation for Weed and Anomaly Detection</h3>
        </div>
        """,
        elem_id="title"
    )
    
    gr.Markdown(
        """
        <div id="description">
        Upload an agricultural field image to detect crops, weeds and anomalies using deep learning.
        Choose from three trained models with different performance characteristics.
        </div>
        """,
        elem_id="description"
    )
    
    # Main content
    with gr.Tabs():
        
        # Tab 1: Single Image Inference
        with gr.Tab(" Single Image Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Field Image",
                        type="pil",
                        height=400
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=list(MODELS.keys())[2],  # Default to Phase 1
                        label="Select Model",
                        info="Choose which model to use for inference"
                    )
                    
                    # Display model info
                    model_info = gr.Markdown(
                        f"**Model Info:** {MODELS[list(MODELS.keys())[2]]['description']}",
                        elem_classes="model-info"
                    )
                    
                    dataset_type = gr.Radio(
                        choices=["WeedMap (3 classes)", "Agriculture-Vision (7 classes)"],
                        value="WeedMap (3 classes)",
                        label="Dataset Type",
                        info="Select the class schema for segmentation"
                    )
                    
                    with gr.Row():
                        opacity_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=60,
                            step=5,
                            label="Overlay Opacity (%)",
                            info="Adjust transparency of segmentation overlay"
                        )
                    
                    show_stats_checkbox = gr.Checkbox(
                        label="Show Class Distribution",
                        value=True,
                        info="Display percentage of each class"
                    )
                    
                    predict_btn = gr.Button("Run Inference", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Segmentation Result",
                        type="pil",
                        height=400
                    )
                    
                    stats_plot = gr.Plot(
                        label="Class Distribution",
                        visible=True
                    )
            
            # Update model info when selection changes
            def update_model_info(model_name):
                return f"**Model Info:** {MODELS[model_name]['description']}"
            
            model_dropdown.change(
                fn=update_model_info,
                inputs=[model_dropdown],
                outputs=[model_info]
            )
            
            # Inference button
            predict_btn.click(
                fn=predict,
                inputs=[input_image, model_dropdown, dataset_type, opacity_slider, show_stats_checkbox],
                outputs=[output_image, stats_plot]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["examples/example1.png", "Phase 1 Finetuned (Agriculture-Vision)", "Agriculture-Vision (7 classes)", 60, True],
                    ["examples/example2.png", "Reproduced RoWeeder (Our Training)", "WeedMap (3 classes)", 70, True],
                    ["examples/example3.png", "Baseline RoWeeder (HuggingFace)", "WeedMap (3 classes)", 50, False],
                ],
                inputs=[input_image, model_dropdown, dataset_type, opacity_slider, show_stats_checkbox],
                outputs=[output_image, stats_plot],
                fn=predict,
                cache_examples=False,
            )
        
        # Tab 2: Model Comparison
        with gr.Tab("Model Comparison"):
            gr.Markdown(
                """
                ### Compare Multiple Models Side-by-Side
                Select 2-3 models to compare their segmentation results on the same image.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    compare_input = gr.Image(
                        label="Upload Field Image",
                        type="pil",
                        height=400
                    )
                    
                    models_checklist = gr.CheckboxGroup(
                        choices=list(MODELS.keys()),
                        value=[
                            "Baseline RoWeeder (HuggingFace)",
                            "Phase 1 Finetuned (Agriculture-Vision)"
                        ],
                        label="Select Models to Compare",
                        info="Choose 2-3 models for comparison"
                    )
                    
                    compare_dataset_type = gr.Radio(
                        choices=["WeedMap (3 classes)", "Agriculture-Vision (7 classes)"],
                        value="WeedMap (3 classes)",
                        label="Dataset Type",
                        info="Select the class schema for segmentation"
                    )
                    
                    compare_opacity = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=60,
                        step=5,
                        label="Overlay Opacity (%)"
                    )
                    
                    compare_btn = gr.Button("Compare Models", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    comparison_output = gr.Image(
                        label="Model Comparison",
                        type="pil",
                        height=600
                    )
            
            compare_btn.click(
                fn=compare_models,
                inputs=[compare_input, models_checklist, compare_dataset_type, compare_opacity],
                outputs=[comparison_output]
            )
        
        # Tab 3: About
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About PRoWeeder
                
                This application demonstrates semantic segmentation for precision agriculture,
                specifically for distinguishing crops from weeds and anomalies in agricultural fields.
                
                ** Key Feature**: Supports both **WeedMap (3 classes)** and **Agriculture-Vision (7 classes)** datasets!
                
                ### Models
                
                **1. Baseline RoWeeder (HuggingFace)**
                - Pretrained model from HuggingFace Hub
                - Performance: 37.5% F1 on WeedMap test set, claimed 75.3% in paper
                - Channels: RGB (3 channels)
                - Output: Always 3 classes (bg, crop, weed)
                - Use case: Baseline comparison
                
                **2. Reproduced RoWeeder (Our Training)**
                - Trained from scratch following the paper methodology
                - Uses crop-row detection pseudo-labels
                - Performance: Target 75.3% F1 on WeedMap
                - Channels: RGB (3 channels)
                - Output: Always 3 classes (bg, crop, weed)
                - Use case: Best for sugar beet fields
                
                **3. Phase 1 Finetuned (Agriculture-Vision)**
                - Finetuned on Agriculture-Vision dataset
                - Performance: 67.0% F1 on Agriculture-Vision, 65.2% F1 on WeedMap
                - Channels: RGB + NIR (4 channels)
                - Output: **7 classes** for Ag-Vision OR **3 classes** for WeedMap
                - Use case: Best for diverse crop types and detailed anomaly detection
                
                ### Dataset Support
                
                **WeedMap (3 Classes):**
                - 🟢 **Green**: Crop (sugar beet)
                - 🔴 **Red**: Weed (unwanted vegetation)
                - ⚫ **Black**: Background (soil, shadows)
                
                **Agriculture-Vision (7 Classes):**
                - ⚫ **Black**: Background
                - 🟢 **Green**: Crop (healthy corn/soy/wheat)
                - 🟠 **Orange**: Planter Skip (missed planting spots)
                - 🔵 **Blue**: Water (standing water in field)
                - 🔴 **Red**: Weed Cluster (weed infestations)
                - 🟡 **Yellow**: Nutrient Deficiency (unhealthy crop areas)
                - 🔵 **Light Blue**: Waterway (drainage channels)
                
                ### How to Use
                
                1. **Select Dataset Type** based on your image:
                   - WeedMap: For sugar beet field images
                   - Agriculture-Vision: For 7 classes field images
                
                2. **Choose Model**:
                   - For WeedMap images: Any model works
                   - For Ag-Vision images: Phase 1 recommended (shows all 7 classes)
                
                3. **Upload Image** and click "Run Inference"
                
                4. **Compare Models** to see the difference in performance!
                
                ### Dataset Information
                
                **WeedMap Dataset:**
                - Source: ETH Zurich & University of Bonn
                - Location: Sugar beet fields in Germany and Switzerland
                - Sensor: Multispectral cameras (RGB + NIR + RedEdge)
                - Resolution: 512x512 pixel tiles
                - Classes: Background, Crop, Weed
                
                **Agriculture-Vision Dataset:**
                - Source: CVPR 2020 Challenge
                - Crops: Corn, soy, wheat
                - Classes: 7 (including weed clusters, waterway, etc.)
                - Scale: Large-scale aerial imagery
                
                ### Performance Comparison
                
                | Model | WeedMap F1 | Ag-Vision F1 | Channels |
                |-------|------------|--------------|----------|
                | Baseline | 37.4% | 2.2% | RGB |
                | Reproduced | ~75% (target) | - | RGB |
                | Phase 1 | 65.2% | 67.0% | RGB+NIR |
                
                ### Technical Details
                
                - **Architecture**: SegFormer-B0 encoder + Custom decoder
                - **Training**: Supervised learning with pseudo-labels (Reproduced) or real labels (Phase 1)
                - **Inference Time**: ~2-3 seconds per image on CPU
                - **Input Size**: 512x512 pixels (auto-resized)
                
                ### Citation
                
                If you use this work, please cite:
                
                ```bibtex
                @inproceedings{de2025roweeder,
                  title={RoWeeder: Unsupervised Weed Mapping through Crop-Row Detection},
                  author={De Marinis, Pasquale and Vessio, Gennaro and Castellano, Giovanna},
                  booktitle={European Conference on Computer Vision},
                  pages={132--145},
                  year={2025},
                  organization={Springer}
                }
                ```
                
                ### Contact
                
                Built for precision agriculture research and education.
                
                ---
                
                Made with ❤️ using Gradio and PyTorch
                """
            )
    
    # Footer
    gr.Markdown(
        """
        <footer>
        <p> PRoWeeeder - Precision Agriculture | 
        <a href="https://arxiv.org/pdf/2410.04983">RoWeeder Paper</a> | 
        <a href="https://github.com/pasqualedem/RoWeeder">RoWeeder Repo</a></p>
        </footer>
        """
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True to create public link
        server_name="127.0.0.1",  # localhost
        server_port=7860,  # Default Gradio port
        show_error=True,
    )