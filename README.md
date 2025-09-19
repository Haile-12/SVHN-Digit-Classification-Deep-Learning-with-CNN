# SVHN Classification with CNN

## 📑 Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
  - [1. Data Loading](#1-data-loading)
  - [2. Preprocessing](#2-preprocessing)
  - [3. CNN Model Architecture](#3-cnn-model-architecture)
  - [4. Training Strategy](#4-training-strategy)
- [Model Evaluation](#model-evaluation)
  - [1. Accuracy](#1-accuracy)
  - [2. Confusion Matrix](#2-confusion-matrix)
  - [3. Classification Report](#3-classification-report)
  - [4. Misclassified Examples](#4-misclassified-examples)
- [Results Summary](#results-summary)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## 📝 Introduction
The objective of this project is to build a **robust digit recognition model** using the **Street View House Numbers (SVHN)** dataset, a real-world dataset composed of digit images from Google Street View.  

Unlike MNIST, SVHN images are in **color** and contain **complex backgrounds**, making it a more challenging and realistic classification problem. Each image is a cropped digit (0–9) from house numbers in street scenes, presented as a **32×32 RGB image**.

### 🌍 Applications include:
- Address and street sign recognition  
- Automated postal sorting  
- OCR in navigation and mapping apps  
- Smart cameras and intelligent transportation systems  

---

## 📊 Dataset Overview
- **Source:** SVHN (Cropped Digits) `.mat` files  
- **Training Samples:** ~73,000  
- **Test Samples:** ~26,000  
- **Classes:** 10 digits (0–9); label `10` is mapped to `0`  
- **Format:** 32×32 RGB images  

### 🔑 Key Characteristics:
- **Complex backgrounds:** Includes background clutter unlike MNIST  
- **Color variance:** Input is 3-channel (RGB)  
- **Real-world noise:** Neighboring digits or lighting issues are present  

---

## ⚙️ Methodology

### 1. Data Loading
Data is loaded from `.mat` files using `scipy.io.loadmat`:  
- `X`: 4D image tensor `(32, 32, 3, N)`  
- `y`: Label vector `(N,)`  

**Preprocessing steps:**  
- Transpose `X` → `(N, 32, 32, 3)`  
- Normalize pixel values to `[0, 1]`  
- Map label `10 → 0`  

---

### 2. Preprocessing
- **One-hot encoding:** Converts numeric labels into categorical vectors.  
- **Train-validation split:** 80/20 split with `train_test_split`.  
- **Data augmentation:** Improves generalization using `ImageDataGenerator` with:  
  - Rotation: ±10°  
  - Zoom: ±10%  
  - Shifts: ±10%  

---

### 3. CNN Model Architecture
A custom **CNN** built with Keras `Sequential` API:  

- **Conv2D layers:** Extract spatial features (filters: 32, 64, 128)  
- **BatchNormalization:** Stabilizes and speeds up training  
- **MaxPooling2D:** Downsamples feature maps  
- **Dropout:** Reduces overfitting  
- **Dense layers:** Map features to class probabilities  
- **Softmax output:** Probability distribution over 10 classes  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  

---

### 4. Training Strategy
- **Batch Size:** 128  
- **Epochs:** 50 (with early stopping)  
- **Callbacks:**  
  - `EarlyStopping` → stops on no validation improvement  
  - `ReduceLROnPlateau` → lowers LR on plateau  

---

## 📈 Model Evaluation

### 1. Accuracy
Achieved **~96% test accuracy**, which is strong given SVHN’s complexity.  

### 2. Confusion Matrix
A 10×10 matrix shows true vs. predicted labels, highlighting:  
- Common misclassifications (e.g., `3 ↔ 5`, `8 ↔ 0`).  

### 3. Classification Report
- **Precision:** Positive predictive value  
- **Recall:** Sensitivity  
- **F1-score:** Harmonic mean of precision & recall  

### 4. Misclassified Examples
Visual inspection shows errors mostly caused by:  
- Blurred/occluded digits  
- Cropping artifacts (neighbor digits remain)  
- Background blending with digits  

---

## 📊 Results Summary
**Digit-wise Performance:**  

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 96%       | 97%    | 96%      |
| 1     | 98%       | 99%    | 99%      |
| 2     | 95%       | 94%    | 95%      |
| 3     | 94%       | 92%    | 93%      |
| 4     | 96%       | 95%    | 95%      |
| 5     | 93%       | 92%    | 93%      |
| 6     | 95%       | 96%    | 96%      |
| 7     | 96%       | 95%    | 96%      |
| 8     | 94%       | 95%    | 95%      |
| 9     | 94%       | 93%    | 94%      |

**Macro Average:**  
- Precision = 0.95  
- Recall = 0.95  
- F1-score = 0.95  

---

## ⚠️ Challenges and Limitations
- **Background clutter** leads to noise in cropped digits  
- **Digit similarity** (e.g., `3 vs 5`, `8 vs 0`) causes confusion  
- **Small variations** (rotation, occlusion) affect predictions  

---

## 🚀 Future Improvements
1. **Leverage Extra Set:** Use the additional 500k labeled samples.  
2. **Transfer Learning:** Try EfficientNet, MobileNet, or ResNet.  
3. **CTC-based Recognition:** Extend to multi-digit sequence recognition.  
4. **Hyperparameter Tuning:** Experiment with LR schedules, dropout, and optimizers.  
5. **Model Ensembling:** Combine multiple CNNs for robustness.  

---

## ✅ Conclusion
This project demonstrates that a carefully designed **CNN** can perform strongly on **real-world digit recognition** tasks with the SVHN dataset.  

- Achieved **~96% accuracy** despite clutter and noise.  
- Shows promise for **address recognition, OCR, and smart vision systems**.  
- With more data and fine-tuning, this model could be deployed in real-world applications.  

---
📌 *Developed as part of a Deep Learning project on Computer Vision.*
