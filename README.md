
# PGA-KD: Principal Geometry Alignment and Semantic Consistency for VLM2Vec Distillation

This is the official PyTorch implementation of the paper:

**"PGA-KD: Principal Geometry Alignment and Semantic Consistency for VLM2Vec Distillation"**

## 📖 Abstract

Adapting large vision–language models (VLMs) for embedding tasks (VLM2Vec) is standard practice, but distilling them into compact architectures is challenging due to mismatched tokenization and "geometric noise" in high-capacity teachers.

**PGA-KD** is a robust distillation framework that addresses these issues via two core components:

1. **Principal Geometry Alignment (PGA):** Decomposes the teacher’s representation space via spectral analysis to filter out high-frequency geometric noise (the "spectral tail"). It forces the student to align only with the principal semantic components ( eigenvectors capturing  energy).
2. **Semantic Consistency Learning (SCL):** Maximizes Mutual Information (MI) across intra-modal (Image-Image, Text-Text) and inter-modal (Image-Text) pathways to ensure the student inherits the teacher's reasoning logic, not just output features.

<p align="center">
<img src="./asset/overview.png" alt="PGA-KD Framework Overview" width="800">





<em>Figure 1: The PGA-KD Distillation Framework.</em>
</p>

## 🛠️ Environment Setup

### 1. Installation

Create a virtual environment and install the required dependencies.

```bash
# Create and activate environment
python -m venv vlm
source vlm/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Library Patch

**Important:** This project requires a modification to the `transformers` library to support Qwen2-VL processing correctly during distillation. Run the provided patch script:

```bash
python fix_lib.py
```

*This script automatically comments out conflicting lines in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`.*

## 📂 Data Preparation

We utilize the **Massive Multimodal Embedding Benchmark (MMEB)** for training and evaluation.

### Automated Download

We provide a script to download the necessary MMEB training and evaluation datasets using `huggingface-hub`. The data will be organized into `./vlm2vec_train/MMEB-train/images`.

```bash
python download.py
```

*Note: This downloads approximately 20 datasets. Please ensure sufficient disk space and a stable internet connection.*

### Manual Download (Optional)

If you only need evaluation images:

```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
rm images.zip
```

## 🚀 Training

All training scripts are located in the `scripts/` directory. We support distillation for two tracks as described in the paper: **FastVLM** (High Compression) and **OneVision** (Architectural Shift).

### 1. Train PGA-KD (Proposed Method)

To train the student model using **Principal Geometry Alignment** and **Semantic Consistency Learning**:

```bash
# Classification Task
bash scripts/cls/train_PGA_fastvlm.sh

bash scripts/cls/train_PGA_llava_onevision.sh

# VQA Task
bash scripts/vqa/train_PGA_fastvlm.sh

bash scripts/vqa/train_PGA_llava_onevision.sh
```
### 2. Train Baselines

We provide scripts to reproduce the baselines reported in the paper (MSE, RKD, CKD, EMO, EM-KD).

```bash
# Standard MSE Distillation for CLS task
bash scripts/cls/train_MSE.sh

# Relational Knowledge Distillation (RKD)
bash scripts/cls/train_RKD.sh

# Comparative Knowledge Distillation (CKD)
bash scripts/cls/train_CKD.sh

# Standard MSE Distillation for VQA task
bash scripts/vqa/train_MSE.sh

# Relational Knowledge Distillation (RKD)
bash scripts/vqa/train_RKD.sh

# Comparative Knowledge Distillation (CKD)
bash scripts/vqa/train_CKD.sh
```

### 3. Configuration

Key hyperparameters (as detailed in the Appendix) can be modified inside the `.sh` scripts:

* `--lambda_pga`: Weight for geometric loss (Default: 2.0)
* `--lambda_scl`: Weight for semantic consistency loss (Default: 0.01)
* `--spectral_threshold`: Cumulative energy threshold  (Default: 0.85)
* `--lora_r`: LoRA rank for the student (Default: 128)

## 📊 Evaluation

To evaluate the distilled model on MMEB (Classification and VQA tasks), run the evaluation script. Ensure the model checkpoint path is correctly set in the script.

```bash
bash scripts/run_eval.sh
```

### Expected Performance

Based on Table 1 and Table 2 of the paper, PGA-KD achieves state-of-the-art results:

| Student Model | Method | Avg Classification | Avg VQA |
| --- | --- | --- | --- |
| **FastVLM-0.5B** | Student (Base) | 64.7 | 50.8 |
|  | RKD | 66.2 | 51.3 |
|  | **PGA-KD (Ours)** | **69.4** | **54.0** |

## 🧩 Model Zoo

| Role | Architecture | Model ID | Dim () |
| --- | --- | --- | --- |
| **Teacher** | Qwen2-VL + SigLIP | `raghavlite/B3_Qwen2_2B` | 1536 |
| **Student 1** | FastVLM + MobileCLIP | `apple/FastVLM-0.5B` | 768 |
| **Student 2** | LLaVA-OneVision | `llava-onevision-qwen2-0.5b-ov-hf` | 768 |

## 🙏 Acknowledgements

We acknowledge the following open-source projects that served as the foundation for our work:

* [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) for the embedding framework and MMEB benchmark.
* [B3-Qwen2](https://github.com/raghavlite/B3) for the teacher model architecture.
* [FastVLM](https://github.com/apple/ml-fastvlm) for the efficient student architecture.