# ❤️ Explainable Multi-Label ECG Classification using 1D CNN on PTB-XL

## 📌 Overview
This project implements an end-to-end deep learning pipeline for multi-label ECG classification using the PTB-XL dataset. A 1D Convolutional Neural Network (CNN) is trained on 12-lead ECG signals to classify multiple cardiac diagnostic superclasses. Grad-CAM-based explainability is integrated to visualize the ECG regions influencing model predictions.

### 🚀 Project Highlights
- 🫀 12-Lead ECG Signal Classification
- 🤖 Deep Learning using TensorFlow/Keras
- 📈 Multi-label Cardiac Disease Prediction
- 🔍 Explainable AI using Grad-CAM
- ⚡ Memory-Optimized ECG Processing Pipeline
- 📊 Comprehensive Evaluation Metrics

---
## 👨‍💻 Author

## Md Affan
Deep Learning & Medical Imaging Enthusiast

---

# 📂 Dataset
Dataset Used:
- **PTB-XL: A Large Publicly Available ECG Dataset**

### 📦 Dataset Contents
- 21,837 clinical 12-lead ECG recordings
- 18,885 patients
- 10-second ECG signals
- 5 diagnostic superclasses

### 🩺 Diagnostic Superclasses
| Label | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

### 📁 Files Used
- `ptbxl_database.csv`
- `scp_statements.csv`
- Low-resolution ECG signal files (`filename_lr`)

---

# 🧠 Project Pipeline

```text
Raw ECG Signals
        ↓
Signal Loading using WFDB
        ↓
Preprocessing & Normalization
        ↓
One-Hot Label Encoding
        ↓
Train / Validation / Test Split
        ↓
1D CNN Model Training
        ↓
Evaluation Metrics
        ↓
Grad-CAM Explainability
```

---

# 📊 Exploratory Data Analysis (EDA)

Performed analyses include:
- 📌 Diagnostic superclass distribution
- ⚖️ Class imbalance visualization
- 🖥️ ECG device distribution
- 👥 Age distribution by ECG device type
- 📈 Sample ECG waveform visualization

### 📚 Visualization Libraries
- Matplotlib
- Seaborn

---

# ⚙️ ECG Signal Preprocessing

## 📥 Signal Loading
ECG signals were loaded using the `wfdb` library.

## ⚡ Resolution Selection
Low-resolution ECG signals (100 Hz) were used to reduce memory usage.

### ECG Shape
```python
(1000, 12)
```

Where:
- `1000` → Time samples
- `12` → ECG leads

## 📏 Normalization
Z-score normalization was applied across ECG leads and time dimensions.

## 🏷️ Label Encoding
Multi-label diagnostic superclasses were one-hot encoded.

---

# ✂️ Dataset Split

| Dataset | Samples |
|---|---|
| Training | 15,285 |
| Validation | 2,184 |
| Testing | 4,368 |

✅ Stratified splitting was used to preserve class distribution.

---

# 🏗️ Model Architecture

The model is a **1D CNN** built using the TensorFlow/Keras Functional API.

### 🔧 Architecture Components
- Conv1D Layers
- Batch Normalization
- MaxPooling1D
- GlobalAveragePooling1D
- Dense Layers
- Dropout Regularization
- Sigmoid Output Layer

### 🎯 Output
Multi-label probabilities for 5 ECG diagnostic superclasses.

---

# 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Epochs | 25 |
| Batch Size | 32 |
| Output Activation | Sigmoid |

### 🛠️ Callbacks Used
- EarlyStopping
- ModelCheckpoint

---

# 📈 Evaluation Metrics

The model was evaluated using:
- ✅ Precision
- ✅ Recall
- ✅ F1-score
- ✅ AUROC
- ✅ ROC Curves
- ✅ Confusion Matrices

## 🏆 Final Results

| Metric | Score |
|---|---|
| Micro F1-score | 0.765 |
| Macro AUROC | 0.927 |

---

# 🔍 Explainable AI using Grad-CAM

Grad-CAM was implemented to improve model interpretability.

### 💡 Grad-CAM Heatmaps
The generated heatmaps:
- Highlight important ECG regions
- Show influential leads and time segments
- Explain model predictions visually

### 🎯 Benefits
- Improved transparency
- Better clinical interpretability
- Increased trustworthiness of predictions

---

# ⚡ Memory Optimization Techniques

To efficiently process the large ECG dataset:
- ✅ Low-resolution ECG signals were used
- ✅ NumPy pre-allocation was implemented
- ✅ In-place normalization was applied
- ✅ `tf.data.Dataset` pipelines were used
- ✅ Prefetching and batching optimized memory usage

These optimizations prevented runtime crashes and reduced RAM consumption.

---

# 💻 Technologies Used

## 👨‍💻 Programming Language
- Python

## 📚 Libraries
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- WFDB
- Scikit-learn

---

# 📁 Project Structure

```text
├── data/
├── notebooks/
├── models/
├── plots/
├── best_cnn_model.keras
├── README.md
```

---

# 🌟 Key Features

- 🫀 Multi-label ECG classification
- 📈 Biomedical signal processing
- 🔬 12-lead ECG analysis
- 🤖 Explainable AI integration
- 🔍 Grad-CAM visualization
- ⚡ Memory-efficient training pipeline
- 🧠 TensorFlow-based implementation

---

# 🚀 Future Improvements

Possible future enhancements:
- ResNet1D architecture
- CNN + BiLSTM hybrid models
- Transformer-based ECG models
- Focal loss for class imbalance
- Advanced ECG signal denoising
- External dataset validation
- Lead-wise explainability analysis

---

# 🏥 Applications

Potential real-world applications:
- Automated cardiac diagnosis
- Clinical decision support systems
- Smart healthcare systems
- Remote ECG monitoring
- AI-assisted cardiology

---

# 📌 Conclusion

This project demonstrates an end-to-end deep learning framework for explainable ECG classification using the PTB-XL dataset. The model achieved strong performance while maintaining interpretability through Grad-CAM visualizations. The pipeline combines biomedical signal processing, efficient data handling, and explainable AI techniques for clinically relevant ECG analysis.

---
