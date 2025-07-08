![](fig/workflow.png)
---
# SRH image classification using ensemble of CLIP, EfficientNet and CatBoost models

This repository contains a ensemble classification of Siberian radioheliograph images that combines EfficientNet (CNN), CLIP (multimodal), and CatBoost (tabular) models.

## [Installation](#-installation) — [Dataset](#-dataset) — [Usage](#-usage) — [Paper](#paper) — [Citation](#citation) — [Contact](#-contact)

## 🧠 Overview

The main components are:

- **`EffnetClassifier`**: Fine-tunes an EfficientNet model for image classification.
- **`ClipClassifier`**: Uses CLIP for zero-shot image classification.
- **`CatBoostTuner`**: Trains a gradient boosting model on extracted embeddings.
- **`EnsembleClassifier`**: Combines predictions from all three models into one final prediction.


Sure! Here's an updated version of your **README** section that includes using a **virtual environment (`venv`)** for better practice and isolation:

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://git.iszf.irk.ru/diegon/srh.git
cd srh

# Create and activate a virtual environment
python -m venv venv         # Create virtual environment
source venv/bin/activate    # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 📥 Dataset

# Create directories
```bash
mkdir -p data
```

# Download the dataset
```bash
wget -O data/3000.zip https://forecasting.iszf.irk.ru/datasets/3000.zip
```

# Extract the dataset
```bash
unzip data/3000.zip -d data/3Ghz
```

# Optional: Remove the zip file
```bash
rm data/3000.zip
```

## 📄 Usage

You can find usage examples in:

```bash
example.ipynb
```

It includes:
- Model evaluation
- Confusion matrix and metrics

## Paper

<!-- Journal Version (Acta Astronautica): https://doi.org/10.1016/j.actaastro.2025.04.027 -->

Open access (arXiv): https://arxiv.org/abs/2507.04211


## Citation

To cite this project, including the scientific basis, models and prepared dataset, please use:

```
@misc{egorov2025srh,
      title={Siberian radioheliograph image classification using ensemble of CLIP, EfficientNet and CatBoost models}, 
      author={Yaroslav Egorov},
      year={2025},
      eprint={2507.04211},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2507.04211}, 
}

```

## 🤝 Contact

Yaroslav Egorov (egorov@iszf.irk.ru)
