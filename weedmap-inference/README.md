---
title: WeedMap AI - Precision Agriculture
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌾 WeedMap AI - Precision Agriculture

Semantic segmentation for weed detection in agricultural fields using deep learning.

## 🎯 Overview

This application demonstrates three different models for crop and weed segmentation:

1. **Baseline RoWeeder (HuggingFace)** - Pretrained model (27.9% F1 on WeedMap)
2. **Reproduced RoWeeder** - Trained from scratch on WeedMap pseudo-labels (Target: 75.3% F1)
3. **Phase 1 Finetuned** - Finetuned on Agriculture-Vision dataset (67.0% F1)

## 🚀 Features

- **Single Image Inference**: Upload any agricultural field image for weed detection
- **Model Comparison**: Compare multiple models side-by-side
- **Interactive Visualization**: Adjust overlay opacity and view class distributions
- **Class Statistics**: See percentage breakdown of crop, weed, and background

## 📊 Model Performance

| Model | WeedMap F1 | Ag-Vision F1 | Channels |
|-------|------------|--------------|----------|
| Baseline | 37.5% | 2.2% | RGB |
| Reproduced | ~75% (target) | - | RGB |
| Phase 1 | 65.2% | 67.0% | RGB+NIR |

## 🎨 Class Labels

- 🟢 **Green**: Crop (sugar beet, corn, soy, wheat)
- 🔴 **Red**: Weed (unwanted vegetation)
- ⚫ **Black**: Background (soil, shadows)

## 📖 Citation

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

## 🔗 Resources

- [RoWeeder Paper](https://arxiv.org/abs/2410.04983)
- [WeedMap Dataset](http://robotics.ethz.ch/~asl-datasets/doku.php?id=weedmap)
- [Agriculture-Vision](https://www.agriculture-vision.com/)

## 📝 License

MIT License - See LICENSE file for details

---

Built with ❤️ using Gradio and PyTorch