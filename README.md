# Coupled Scene Graph Anticipation (CoupledSGA)

Official PyTorch implementation of the paper:  
**"Coupled Scene Graph Anticipation: When With-Constraint Recall Hides Structural Inconsistency"**  
*Accepted at ICVGIP 2026.*

---

## 📌 Overview

Scene Graph Anticipation (SGA) aims to forecast future human-object interactions and spatial layouts from partial video observations. However, standard evaluation under **With-Constraint Recall** forces a winner-take-all selection per relationship category, which **masks severe underlying structural contradictions**—such as models simultaneously assigning high confidence to physically impossible predicate pairs (e.g., *above* / *beneath*, *holding* / *not_contacting*).

**CoupledSGA** introduces:
1. **Unconstrained Structural Evaluation:** Evaluating SGA under **No-Constraint Recall**, paired with **Contradiction Rate (CR)** and **Co-Activation Mass (CAM)** to diagnose structural conflicts.
2. **Coupled Graph Dynamics:** A continuous-discrete neural ODE formulation coupling object node state evolutions with dynamic edge attention.
3. **Curriculum-Scheduled Structural Consistency ($\mathcal{L}_{\mathrm{SC}}$):** An exclusionary regularizer applied with a delayed curriculum start ($e_{\mathrm{start}} = 2$, $\lambda_{\mathrm{SC}} = 0.1$) that drastically suppresses contradictory co-activations while preserving anticipatory recall.

---

## 📂 Repository Structure

```text
CoupledSGA_Repo/
├── dataloader/                   # Action Genome data loaders and dataset parsers
│   └── action_genome/
│       └── ag_dataset.py
├── datasets/                     # Data processing utilities
│   └── action_genome/tools/
│       ├── build_exact_split_video_ids.py
│       ├── dump_frames.py
│       ├── filter_action_genome_subset.py
│       └── load_annotations.py
├── fasterRCNN/                   # Pretrained Faster R-CNN backbone and ROI layers
│   ├── lib/                      # Faster R-CNN CUDA/C extensions
│   ├── models/
│   │   └── model.txt             # Download link for faster_rcnn_ag.pth
│   ├── test_net.py
│   └── trainval_net.py
├── lib/                          # Core neural architecture and model definitions
│   ├── supervised/
│   │   ├── sga/
│   │   │   ├── coupled_sga.py                  # CoupledSGA architecture
│   │   │   ├── structural_consistency_loss.py  # L_SC loss formulation
│   │   │   ├── scene_sayer_ode.py              # SceneSayer ODE baseline
│   │   │   ├── scene_sayer_sde.py              # SceneSayer SDE baseline
│   │   │   ├── sttran_ant.py                   # STTran+ baseline
│   │   │   └── dsgdetr_ant.py                  # DSG-DETR+ baseline
│   │   └── evaluation_recall.py                # Recall evaluation (With/No-Constraint)
│   ├── fpn/box_intersections_cpu/              # Bounding box intersection C-extension
│   └── draw_rectangles/                        # Visualization C-extension
├── rebuttal_results/             # Rebuttal evaluation logs & prediction tensors
│   ├── predictions/
│   │   ├── A.npz                 # Baseline (SceneSayer ODE) predictions
│   │   ├── B.npz                 # CoupledSGA (λ=0, no L_SC) predictions
│   │   └── C.npz                 # CoupledSGA (λ=0.1, estart=2) predictions
│   ├── results.json              # Tabular comparison summary
│   └── run_log.md                # Verification and reproduction log
├── results/                      # Evaluation output CSVs
│   └── predcls/                  # Benchmark logs across observation ratios (τ ∈ {0.3, 0.5, 0.7, 0.9})
├── scripts/                      # Training and evaluation shell/batch scripts
│   ├── train.txt                 # Training command lines across modes (predcls, sgcls, sgdet)
│   └── test.txt                  # Testing command lines
├── constants.py                  # Global constants and Action Genome categories
├── logger_config.py              # Logging setup
├── sga_base.py                   # SGA Base classes
├── structural_metrics.py         # CR, CAM, and SCS metric computation script
├── test_structural_metrics.py    # Unit tests for structural metrics
├── train_sga_base.py             # Training loop base implementation
├── train_sga_methods.py          # SGA training entrypoint
├── test_sga_base.py              # Evaluation loop base implementation
├── test_sga_methods.py           # SGA testing entrypoint
├── video_800_file.txt            # Action Genome 800-video subset ID list
├── video_800_file_train.txt      # 600 training video IDs (Footnote 1 split)
├── video_800_file_test.txt       # 200 testing video IDs (Footnote 1 split)
├── video_50_file.txt             # 50-video debug subset ID list
├── video_50_file_train.txt       # 45 debug training video IDs
├── video_50_file_test.txt        # 5 debug testing video IDs
├── requirements.txt              # Python package dependencies
├── LICENSE                       # MIT License
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Create Environment
```bash
conda create -n coupledsga python=3.10 -y
conda activate coupledsga
```

### 2. Install PyTorch & Dependencies
```bash
# Install PyTorch with appropriate CUDA support, e.g.:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install required Python packages
pip install -r requirements.txt
```

### 3. Build C/Cython Extensions
Compile the bounding box and rectangle drawing extensions:

```bash
# 1. Build box_intersections_cpu
cd lib/fpn/box_intersections_cpu
python setup.py build_ext --inplace
cd ../../../

# 2. Build draw_rectangles
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ../../

# 3. Build Faster R-CNN CUDA/C++ extensions
cd fasterRCNN/lib
python setup.py build develop
cd ../../
```

---

## 📥 Pretrained Weights & Downloads

Because pretrained checkpoints and model weights exceed GitHub's file storage limits, download them from the links below and place them in their respective directories:

### 1. Faster R-CNN Feature Backbone
- **File:** `faster_rcnn_ag.pth` (~362 MB)
- **Download Link:** [Box Download (Official)](https://utdallas.box.com/s/1pspm5x8etlczoklyw4bclotsmlejagj)
- **Destination:** Place in `fasterRCNN/models/faster_rcnn_ag.pth`
*(Note: Download instructions are also documented inside `fasterRCNN/models/model.txt`)*

### 2. CoupledSGA Model Checkpoints (Ours)
Download our trained checkpoints for reproduction and place them inside `checkpoints/`:
- **Model C (CoupledSGA with $\mathcal{L}_{\mathrm{SC}}$, $\lambda=0.1$):** `checkpoints/model_c_sc/model_c_sc_epoch_3.tar`
- **Model B (CoupledSGA unregularized, $\lambda=0$):** `checkpoints/model_b_nosc/model_b_nosc_epoch_3.tar`
- **Model A (Baseline SceneSayer ODE):** `checkpoints/ode_predcls_future_5/ode_predcls_future_5_epoch_3.tar`
- **Download Link:** [Google Drive Download Link](https://drive.google.com/) *(Add your shareable Google Drive link here)*

### 3. Baseline SGA Checkpoints (Prior Work)
Download baseline checkpoints from the SceneSayer repository and organize them inside `checkpoints/`:
- **Action Genome Scenes (AGS / sgdet):** [Download Link](https://utdallas.box.com/s/xaf2ifdla5y1qm7prh175d6e3t3yoz79)
- **Partially Grounded Action Genome Scenes (PGAGS / sgcls):** [Download Link](https://utdallas.box.com/s/qomw52816xiylln15j64nfj5lhg7d56t)
- **Grounded Action Genome Scenes (GAGS / predcls):** [Download Link](https://utdallas.box.com/s/py5yt40pt3bokqbyx7wc0o2imzn3j5tq)

Checkpoint naming follows:
```text
<CKPT_METHOD_NAME>_<MODE>_future_<#FUTURE_FRAMES>_epoch_<#EPOCH>.tar
```

---

## 📊 Dataset Preparation

1. Download the [Action Genome dataset](https://github.com/JingweiJ/ActionGenome) annotations and [Charades](https://allenai.org/data/charades) videos:
   - Videos: `data/ag/videos`
   - Annotations: `data/ag/annotations`
   - Extracted Frames: `data/ag/frames`
2. Download `object_bbox_and_relationship_filtersmall.pkl` from [here](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view) and place in `dataloader/`.
3. **Partition Splits:** The exact 800-video subset evaluated in the paper is partitioned into 600 training videos and 200 testing videos:
   - Training IDs: `video_800_file_train.txt`
   - Testing IDs: `video_800_file_test.txt`

---

## 🚀 Training CoupledSGA

Train CoupledSGA using the curriculum schedule ($\lambda_{\mathrm{SC}} = 0.1$, $e_{\mathrm{start}} = 2$):

```bash
# Grounded Action Genome Scenes (predcls) - Anticipating 5 future frames
python train_sga_methods.py \
    --method_name=coupledsga \
    --mode=predcls \
    --max_window=5 \
    --lambda_sc=0.1 \
    --sc_start_epoch=2 \
    --bce_loss

# Partially Grounded Action Genome Scenes (sgcls)
python train_sga_methods.py \
    --method_name=coupledsga \
    --mode=sgcls \
    --max_window=5 \
    --lambda_sc=0.1 \
    --sc_start_epoch=2 \
    --bce_loss

# Action Genome Scenes (sgdet)
python train_sga_methods.py \
    --method_name=coupledsga \
    --mode=sgdet \
    --max_window=5 \
    --lambda_sc=0.1 \
    --sc_start_epoch=2 \
    --bce_loss
```

Additional training commands across window sizes ($T=1, 3, 5$) are provided in `scripts/train.txt`.

---

## 🔍 Evaluation & Structural Metrics

### 1. Evaluate Anticipation Performance
To evaluate a trained checkpoint across observation ratios $\tau \in \{0.3, 0.5, 0.7, 0.9\}$:

```bash
python test_sga_methods.py \
    --method_name=coupledsga \
    --mode=predcls \
    --ckpt=checkpoints/coupledsga_predcls_future_5_epoch_3.tar
```

### 2. Reproduce Structural Consistency Metrics (CR, CAM, SCS)
To compute Contradiction Rate (CR) directly from saved prediction arrays (without requiring dataset downloads):

```bash
python structural_metrics.py
```

Expected output matching the paper:
```text
=======================================================
Model      | CR (%)     | CAM        | SCS (%)   
-------------------------------------------------------
Model A    | 54.40      | 17598.80   | 100.00    
Model B    | 45.92      | 15830.75   | 100.00    
Model C    | 4.07       | 14513.30   | 100.00    
=======================================================
```
* **Model A:** Baseline (SceneSayer ODE) — high contradiction rate (54.40%).
* **Model B:** CoupledSGA ($\lambda_{\mathrm{SC}} = 0$, unregularized).
* **Model C:** CoupledSGA ($\lambda_{\mathrm{SC}} = 0.1, e_{\mathrm{start}} = 2$) — contradictions drop to **4.07%** (>92% relative reduction).

### 3. Run Metric Unit Tests
```bash
python test_structural_metrics.py
```

---

## 📜 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@inproceedings{coupled_sga_2026,
  title     = {Coupled Scene Graph Anticipation: When With-Constraint Recall Hides Structural Inconsistency},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year      = {2026}
}
```

---

## 📄 License
This repository is released under the [MIT License](LICENSE).
