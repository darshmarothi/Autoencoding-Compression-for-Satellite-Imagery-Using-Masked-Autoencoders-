# Autoencoding-Compression-for-Satellite-Imagery-Using-Masked-Autoencoders-
 This project is a milestone on the way of realizing of Light-weight Masked Autoencoders as a viable paradigm for  satellite image compression. By acquiring compact, semantically meaningful latent representations, the MAE model makes  the trade-off between compression ratio and reconstruction fidelity better. 

This project implements a **Lightweight Masked Autoencoder (MAE)** for **satellite image compression**, evaluated against a **PCA baseline** using the RGB EuroSAT dataset.

---

## 🚀 Key Features
- Deep-learning-based compression using a **Transformer MAE**
- **75% masked patches** for self-supervised learning
- Latent space size: **16 KB per image**
- PCA baseline configured for matched storage
- Metrics:
  - PSNR
  - SSIM
  - SAM
- **Streamlit Web App** for:
  - Uploading satellite image  
  - Compression → Reconstruction  
  - Downloading latent vectors  
  - Upload latent → Reconstruct image  

---
## Model Flow
<img width="900" height="441" alt="image" src="https://github.com/user-attachments/assets/8cc398ea-650f-45d6-a0ea-92d38b67f8f2" />

- Figure 1. MAE Workflow for Image Compression
---

## 📚 Project Structure
Satellite-MAE-Compression/
│
├── README.md
│
├── project_report/
│   └── projectreportadctfinal.pdf
│
├── notebooks/
│   └── MaeVsPCA.ipynb
│
├── src/
│   ├── mae_model.py
│   ├── train_mae.py
│   ├── evaluate.py
│   └── utils.py
│
├── streamlit_app/
│   └── app.py
│
├── saved_models/
│   └── mae_light_corrected_final.pth
│
└── requirements.txt


---

## 🧠 Model Summary
- Input: **128×128 RGB satellite image**
- Patch size: **16×16**
- Total patches: 64
- Masking: **75%**
- Encoder processes **16 visible patches**
- Decoder reconstructs all patches
- Latent vector storage: **16 KB**

---

## 📊 Results Summary
| Method | Size (bytes) | PSNR | SSIM | SAM |
|--------|--------------|------|------|------|
| PCA (k=1) | ~65,560 | 32.496 | 0.986 | 0.0397 |
| MAE       | ~16,384 | 24.777 | 0.844 | 0.0307 |

---

## ▶ Running the Streamlit App
```
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

---

## 🔗 References
- Masked Autoencoders (He et al., CVPR 2022)
- EuroSAT Dataset
- Transformer Encoder Architecture

---

## 👤 Contributors
- Aman N Shah – 22MID0250  
- Darsh Marothi – 22MID0333  
- Trisha – 22MID0263  

