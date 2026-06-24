# ❤️ Heart Disease Prediction

A Machine Learning project for predicting the presence of heart disease using clinical patient data. This project compares the performance of a traditional Machine Learning model (**Logistic Regression**) with a Deep Learning model (**Feed Forward Neural Network - FFN**).

---

## 📌 Project Overview

The goal of this project is to build and evaluate classification models that can predict whether a patient has heart disease based on medical attributes such as age, cholesterol level, chest pain type, and other clinical measurements.

Two models were implemented and compared:

- Logistic Regression
- Feed Forward Neural Network (PyTorch)

---

## 📊 Dataset

**Heart Disease Dataset (Cleveland)**

Features include:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression (Oldpeak)
- Slope
- Number of Major Vessels (ca)
- Thalassemia (thal)

Target:

- `0` → No Heart Disease
- `1` → Heart Disease

---

## ⚙️ Data Preprocessing

- Train / Validation / Test split
- Stratified sampling
- Standardization of numerical features
- One-Hot Encoding of categorical features
- Threshold optimization based on Validation F1-Score

---

## 🤖 Models

### Logistic Regression

- LogisticRegressionCV
- Cross-validation for hyperparameter selection
- Threshold tuning using Validation F1-score

### Feed Forward Neural Network (FFN)

Architecture:

```text
Input Layer
     ↓
Linear(Features → 32)
     ↓
LeakyReLU
     ↓
Dropout(0.3)
     ↓
Linear(32 → 16)
     ↓
LeakyReLU
     ↓
Dropout(0.3)
     ↓
Linear(16 → 2)
```

Training:

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Early model selection based on Validation Accuracy

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.94 | 0.93 | 0.93 | 0.93 | 0.98 |
| FFN | 0.90 | 0.91 | 0.91 | 0.90 | 1.00 |

### Key Observation

- Logistic Regression achieved the highest F1-score and overall classification performance.
- FFN achieved a perfect ROC-AUC score on the test split but slightly lower F1-score.
- For this dataset size, Logistic Regression performed competitively against the neural network while requiring significantly less complexity.

---

## 📉 ROC Curve Comparison

<table>
<tr>
<th align="center">Logistic Regression</th>
<th align="center">Feed Forward Neural Network (FFN)</th>
</tr>

<tr>
<td align="center">
<img src="RocCurveDisplay_logestic.png" width="450">
</td>

<td align="center">
<img src="RocCurveDisplay.png" width="450">
</td>
</tr>
</table>

---

## 📂 Project Structure

```text
Heart_Disease/
│
├── FFN_Model/
│   ├── build_model.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.py
│
├── Logestic_Regression/
│   ├── preprocessing.py
│   ├── training_logestic.py
│   ├── evaluation_logestic.py
│   └── main_logestic.py
│
├── Heart Disease Dataset_EDA.ipynb
├── processed.cleveland.data
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git
```

Move to the project directory:

```bash
cd ML_Portfolio/Machine_Learning_Project/Heart_Disease
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- PyTorch

---

## 👩‍💻 Author

**Ziba**

- MSc in Biotechnology
- Machine Learning & Deep Learning Enthusiast
- Interested in AI applications in healthcare

GitHub:
https://github.com/zziibbaa

---
