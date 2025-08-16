# Interpretable Alzheimer’s Detection from 3D MRI with OASIS-3

## Project Overview

This project tackles one of the most urgent challenges in modern healthcare: **early and trustworthy detection of Alzheimer’s disease (AD)** using MRI brain scans. While deep learning models can achieve high accuracy in medical imaging, **clinical adoption is hampered by the “black box” problem**—clinicians cannot trust models if they do not understand *why* a decision was made.  

Our work addresses this barrier head-on by integrating **Explainable AI (XAI)** into a deep learning pipeline. We not only classify MRI scans into healthy vs. demented categories but also **show which brain regions drove the decision**, mapping model activations onto the **Harvard–Oxford Cortical Atlas**.  

---

## Why Explainability Matters

- Alzheimer’s disease affects **55+ million people worldwide** and cases are expected to double every 20 years.  
- Deep learning models often achieve high accuracy, but without interpretability, **clinicians resist adoption**.  
- Prior healthcare AI failures (IBM Watson for Oncology, DeepMind Streams, etc.) highlight that **lack of trust kills adoption**, even when accuracy is high.  
- Our solution integrates **3D Grad-CAM** with neuroanatomical mapping to make the AI’s decision-making transparent and clinically relevant.  

---

## Dataset: OASIS‑3 (Longitudinal MRI)

**OASIS‑3** is one of the largest open neuroimaging datasets, including over **2,000 MRI sessions from 1,000+ participants**, with longitudinal follow-up and clinical diagnoses.  

- **T1‑weighted MRI scans** were selected for classification.  
- Labels are derived from **Clinical Dementia Rating (CDR)** scores:  
  - **CDR = 0 → Healthy (label 0)**  
  - **CDR = 0.5 → Mild Cognitive Impairment (MCI, label 1)**  
  - **CDR ≥ 1.0 → Alzheimer’s Disease (AD, label 2)**  

**Final curated subset (1 scan per subject):**  
- Healthy: 829  
- MCI: 232  
- AD: 81  

### How to Access OASIS‑3
1. **Request Access**: Go to the [OASIS‑Brains portal](https://www.oasis-brains.org) and create an account.  
2. **Sign the Data Use Agreement (DUA)** to gain approval.  
3. **Log in to the OASIS‑3 XNAT platform** after approval.  
4. Download **T1-weighted MRI scans** in NIfTI format.  
5. Use the official **`oasis_data_matchup.R`** script to align MRI sessions with CDR scores.  
6. Select **one scan per subject** (closest to baseline).  

**Preprocessing applied:**  
- Resample to `(128×128×128)` voxels.  
- Z-score intensity normalization.  
- Augmentations: random flips, Gaussian noise, intensity jitter.  

---

## Methodology

### Model
- Backbone: **3D ResNet-34 / 3D ResNet-50** (MONAI, pretrained on MedicalNet).  
- Head: Flatten → Linear (2-class or 3-class).  
- Loss: **Class-weighted Cross-Entropy** to address imbalance.  
- Training: 10–20 epochs, batch size = 2, stratified 80/20 split.  
- Fine-tuning: progressively unfreeze `layer3`, `layer4`, and `fc`.  
- Prioritized metric: **macro recall (sensitivity)**—critical in medicine to avoid missed diagnoses.  

### Explainability (XAI)
1. Compute **3D Grad‑CAM** from the last convolutional block.  
2. Upsample to input resolution `(128³)`.  
3. Convert to NIfTI and overlay with **Harvard–Oxford Cortical Atlas**.  
4. Quantify overlap per brain region for **clinically interpretable reports**.  

---

## Results

**Best binary model (Healthy vs Demented):**  
- Test Recall: **0.9365 – 0.9841** (depending on run)  
- Test F1 Score: ~0.63  
- Test Precision: ~0.47  
- Test Accuracy: ~0.53  

**Key Findings:**  
- The model focuses on **hippocampus, medial temporal lobe, frontal pole, and anterior cingulate**, all clinically validated biomarkers for AD.  
- Some regions (e.g., parahippocampal gyrus) were under-activated—highlighting areas for improvement and future multimodal integration.  

---

## How to Reproduce

### 1) Environment
```bash
pip install -U torch torchvision torchaudio
pip install -U monai nibabel nilearn numpy scipy scikit-learn matplotlib
pip install -U wandb   # optional for experiment tracking
```

### 2) Data Preparation
```
data/
  train/
    healthy/ *.nii.gz
    demented/ *.nii.gz
  test/
    healthy/ *.nii.gz
    demented/ *.nii.gz
```

### 3) Training Example
```python
model = build_3d_resnet(num_classes=2)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

for epoch in range(1, 16):
    train_one_epoch(...)
    metrics = evaluate(...)
    save_if_best(metrics["recall_macro"], model)
```

### 4) Explainability
```python
heatmap = gradcam_3d(model, volume)
report  = atlas_overlap(heatmap, atlas)
show_overlays(volume, heatmap)
```

---

## Repository Structure
```
.
├── GoTG_FinalProject_OASIS_3.ipynb       # full pipeline
├── FinalProjectReport_GoTG.docx          # report (methods & results)
├── src/
│   ├── data.py        # preprocessing & transforms
│   ├── model.py       # 3D ResNet definitions
│   ├── train.py       # training & evaluation
│   └── xai.py         # 3D Grad-CAM + atlas mapping
├── atlas/             # Harvard–Oxford atlas files
├── runs/              # saved checkpoints
└── README.md
```

---

## Ethics & Compliance
- Data used strictly under **OASIS‑3 Data Use Agreement**.  
- Project intended for **research purposes only**.  
- **Not for clinical use** without IRB approval and extensive validation.  
- XAI provides **interpretive support**, not diagnostic ground truth.  

---

## Acknowledgements
- **OASIS‑3 investigators and participants**  
- **Harvard–Oxford Cortical Atlas** (FSL)  
- **MONAI** (Medical Open Network for AI)  
- Guardians of the Galaxy Team (ISM6561 Project, USF)  

---

## Citation
If you use this work, please cite our final project report:  
*“Beyond the Black Box: A Story of Trust, Transparency, and Technology in Alzheimer’s Detection” (Guardians of the Galaxy, USF, 2025).*  
