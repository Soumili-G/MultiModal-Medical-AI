# MultiModal-Medical-AI

### AI-Powered Medical Image Analysis and Decision-Support System

MultiModal-Medical-AI is a deep-learning-based medical image analysis project designed to process and analyze different types of medical imaging data through a unified application.

The system integrates six ResNet18-based models covering MRI, chest X-ray, and four ultrasound applications. It provides image classification, multi-label disease prediction, fetal AOP angle regression, and Grad-CAM visual explanations through an interactive Streamlit interface.

> **Disclaimer:** This project is an experimental academic research prototype. It is not a clinically validated medical device and must not be used for diagnosis, treatment decisions, or emergency medical assessment.

---

## 1. Project Overview

Medical imaging plays an important role in identifying and evaluating different medical conditions. However, different imaging modalities require specialized analysis models.

This project brings multiple imaging pipelines together into one application.

Users can:
- Select an imaging modality.
- Select an ultrasound subtype when applicable.
- Upload a medical image.
- Receive model-generated predictions.
- View prediction confidence or disease scores.
- Visualize model attention using Grad-CAM overlays.

### Supported modalities

| Modality | Model | Task |
|---|---|---|
| MRI | ResNet18 | 4-class classification |
| Chest X-ray | ResNet18 | 14-label classification |
| Breast ultrasound | ResNet18 | 3-class classification |
| Ovarian ultrasound | ResNet18 | 5-class classification |
| Abdominal ultrasound | ResNet18 | 10-class classification |
| Fetal ultrasound | ResNet18 | AOP angle regression |

---

## 2. System Architecture

The application follows a modular architecture.

1. **User Interface:** Streamlit handles image uploads and displays results.
2. **Image Validation:** CLIP-based screening checks whether an uploaded image resembles the selected medical imaging modality.
3. **Model Router:** Selects the appropriate model based on the user's modality and subtype.
4. **Inference Modules:** Preprocess images and generate predictions.
5. **Grad-CAM:** Generates visual explanations from the model's feature activations and gradients.
6. **Results Display:** Shows predictions, scores, and original/Grad-CAM overlay images.

### Workflow

```text
                 USER
                   |
                   v
            STREAMLIT APP
                   |
                   v
             IMAGE UPLOAD
                   |
                   v
           IMAGE VALIDATION
                   |
          +--------+--------+
          |                 |
       Rejected           Accepted
          |                 |
          v                 v
       Error            MODEL ROUTER
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
         MRI              X-RAY             ULTRASOUND
          |                 |                  |
          v                 v          +-------+-------+
       ResNet18          ResNet18       |       |       |
                                       Breast  Ovarian Abdominal/Fetal
                                        |       |       |
                                        v       v       v
                                      ResNet18 Models
          |                 |                  |
          +-----------------+------------------+
                            |
                            v
                       PREDICTION
                            |
                            v
                         GRAD-CAM
                            |
                            v
                     RESULTS DISPLAY
```

---

## 3. Model Details

### A. MRI Classification

**Architecture:** ResNet18

**Classes:**
- Glioma
- Meningioma
- No Tumor
- Pituitary

**Task:** Multi-class brain MRI image classification.

**Checkpoint:**
`models/mri_model/best_mri_model.pt`

### B. Chest X-ray Classification

**Architecture:** ResNet18

**Task:** Multi-label chest X-ray classification.

The model predicts scores for 14 findings:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

**Checkpoint:**
`models/xray_model/resnet18/best_resnet18.pth`

The model uses sigmoid outputs because multiple findings may be present in the same image.

**Reported evaluation results from the X-ray training run:**

| Metric | Result |
|---|---:|
| Validation Macro AUC | Approximately 0.816 |
| Test Macro AUC | Approximately 0.788 |
| Test Micro AUC | Approximately 0.834 |

These are experimental dataset evaluation results, not clinical performance guarantees.

### C. Breast Ultrasound Classification

**Architecture:** ResNet18

**Classes:**
- Benign
- Malignant
- Normal

**Checkpoint:**
`models/ultrasound_models/breast_model.pt`

### D. Ovarian Ultrasound Classification

**Architecture:** ResNet18

**Classes:**
- Complex Cyst
- Dominant Follicle
- Healthy
- Polycystic
- Simple Cyst

**Checkpoint:**
`models/ultrasound_models/ovarian_model.pt`

### E. Abdominal Ultrasound Classification

**Architecture:** ResNet18

**Classes:**
- AA
- GB
- Hepatic
- Kidney
- Liver
- Ovaries
- Pancreas
- Portal
- Spleen
- UB + Prostate + Uterus + Cervix

**Checkpoint:**
`models/ultrasound_models/abdominal_model.pt`

### F. Fetal Ultrasound AOP Regression

**Architecture:** ResNet18

**Task:** Regression-based prediction of the fetal Angle of Progression (AOP).

Unlike the classification models, this model produces a continuous numerical output representing an estimated angle in degrees.

**Checkpoint:**
`models/ultrasound_models/fetal_aop_model.pt`

The prediction is converted back from normalized model output using the mean and standard deviation stored in the checkpoint.

---

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Model Architecture | ResNet18 |
| Image Validation | CLIP, Hugging Face Transformers |
| Explainable AI | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Machine Learning Utilities | Scikit-learn |
| User Interface | Streamlit |
| Development Environment | VS Code, Jupyter Notebook |
| Training Environment | Kaggle GPU |

---

## 5. Project Structure

```text
MultiModal-Medical-AI/
│
├── app/
│   ├── streamlit_app.py
│   └── router.py
│
├── datasets/
│   ├── mri/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── metadata.csv
│   │
│   ├── xray/
│   │   ├── raw/
│   │   └── metadata.csv
│   │
│   └── ultrasound/
│       ├── raw/
│       │   ├── busi/
│       │   ├── ovarian/
│       │   ├── abdominal/
│       │   └── fetal/
│       └── processed/
│
├── models/
│   ├── mri_model/
│   │   └── best_mri_model.pt
│   │
│   ├── xray_model/
│   │   └── resnet18/
│   │       ├── best_resnet18.pth
│   │       └── latest_resnet18.pth
│   │
│   └── ultrasound_models/
│       ├── breast_model.pt
│       ├── ovarian_model.pt
│       ├── abdominal_model.pt
│       └── fetal_aop_model.pt
│
├── training/
│   ├── train_mri.py
│   ├── train_ultrasound.py
│   ├── train_fetal_aop.py
│   └── evaluation scripts
│
├── utils/
│   ├── mri_inference.py
│   ├── xray_inference.py
│   ├── ultrasound_inference.py
│   ├── gradcam.py
│   ├── image_validation.py
│   ├── dataset.py
│   └── evaluation.py
│
├── results/
├── outputs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

*The tree describes the main project organization. Some dataset, results, and training files may vary according to the local setup.*

---

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or later, compatible with the installed PyTorch and torchvision versions.
- Visual Studio Code.
- Git.
- Sufficient storage for model checkpoints and datasets.
- Internet access for the initial CLIP model download.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd MultiModal-Medical-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
```

If `requirements.txt` is present:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Verify PyTorch

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The application supports CPU inference and can use CUDA when a compatible GPU environment is available.

### Step 5: Launch the application

From the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

---

## 7. Using the Application

1. Launch the Streamlit application.
2. Select MRI, X-ray, or Ultrasound.
3. If Ultrasound is selected, choose the relevant subtype.
4. Upload an image in a supported format.
5. Click **Analyze**.
6. Review the model-generated prediction and scores.
7. Examine the original image alongside the Grad-CAM overlay.

### Image Validation

The application uses CLIP-based image screening to compare uploaded images with descriptions of medical scans and unrelated image types.

This is an initial screening mechanism, not a validated medical-image detector. It may incorrectly accept or reject images.

### Grad-CAM Visualization

Grad-CAM uses gradients and intermediate convolutional feature maps to generate a heatmap indicating image regions that influenced a model's selected output.

The application displays:
- Original uploaded image.
- Grad-CAM overlay.

Grad-CAM visualizations are approximate explanations and do not establish that a prediction is correct or that the highlighted region is medically meaningful.

---

## 8. Datasets

The project uses publicly available medical imaging datasets.

| Modality | Dataset |
|---|---|
| MRI | Brain Tumor MRI dataset |
| Chest X-ray | NIH Chest X-ray dataset |
| Breast Ultrasound | BUSI dataset |
| Ovarian Ultrasound | Ovarian ultrasound dataset |
| Abdominal Ultrasound | Abdominal organ ultrasound dataset |
| Fetal Ultrasound | Fetal ultrasound dataset for AOP regression |

Dataset access, licensing, and attribution requirements should be checked before redistribution.

**Note:** Dataset images are not included in this README. Download and organize the datasets separately according to the project directory structure.

---

## 9. Model Evaluation

Different tasks require different evaluation metrics.

| Model Type | Relevant Evaluation Metrics |
|---|---|
| Multi-class classification | Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC |
| Multi-label classification | Macro AUC, Micro AUC, Per-class AUC, Precision, Recall, F1-score |
| Regression | MAE, RMSE, R² |

For multi-label X-ray classification, ROC-AUC is particularly useful because each disease label is evaluated separately.

For fetal AOP regression, MAE and RMSE measure the difference between predicted and reference angles.

Not all evaluation results are currently reported in this README. Additional results should be added only after they have been measured on the appropriate held-out datasets.

---

## 10. Important Limitations

- The models are experimental and have not been clinically validated.
- Dataset imbalance can affect prediction performance.
- Predictions may vary with image quality, acquisition equipment, and dataset distribution.
- A high confidence score does not necessarily indicate a correct prediction.
- CLIP-based screening is heuristic and is not a guarantee that an image is a valid medical scan.
- Grad-CAM is an approximate visualization, not a clinical explanation.
- The models should not replace qualified medical professionals.

---

## 11. Future Improvements

Potential future development includes:

- More extensive external validation.
- Improved out-of-distribution image detection.
- Calibration of classification probabilities.
- Per-class performance reports for all classification models.
- Regression error analysis for fetal AOP.
- Improved model explainability.
- Additional medical imaging modalities.
- Model optimization for faster inference.
- Secure deployment and user-data protection.

---

## 12. Author

**MultiModal-Medical-AI**

An academic deep-learning project integrating multiple medical imaging modalities into a unified AI-assisted image analysis application.

---

## 13. License

Add the appropriate license after confirming the licenses and redistribution conditions of the code and datasets used in the project.