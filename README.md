# traffic-sign
# 🚦 Traffic Sign Classification Project

This project focuses on building a high-accuracy deep learning model for traffic sign recognition using the GTSRB dataset. The model is trained from scratch and optimized for real-world usage such as mobile or embedded applications.

---

## 📌 Included in This Commit

### ✔️ 1. Model & Source Code
This commit introduces the core training and testing infrastructure:

- **train_model.py** – Main training script  
- **train_model_test.py** – Evaluation / testing script  
- **TrafficSignModel_best.keras** – Best-performing model weights  

Model performance summary:

| Metric | Value |
|--------|--------|
| **Training Accuracy** | 0.98+ |
| **Validation Accuracy** | 0.99+ |
| **Test Accuracy** | 0.97+ |

---

### ✔️ 2. Documentation
Included documentation file:

- **TrafficSingModelBilgisi.docx**  
  - Model architecture  
  - Hyperparameters  
  - Training logs  
  - Evaluation results  
  - Usage notes  

---

### ✔️ 3. Excluded From This Commit
To keep the repository lightweight, the following large folders are ignored:

- `Meta/`
- `Train/`
- `Test/`

These directories contain the dataset and are handled via `.gitignore`.

---

## 🎯 Project Goal

To build a robust traffic sign classifier capable of accurately predicting traffic sign categories from images, suitable for real-world deployment such as autonomous systems, detection pipelines, or mobile apps.

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV (optional)  
- GTSRB Dataset  

---

## 📁 Repository Structure (for this commit)

```
project-root/
│
├── train_model.py
├── train_model_test.py
├── TrafficSignModel_best.keras
├── TrafficSingModelBilgisi.docx
├── README.md
└── .gitignore
```

---

## ▶️ Quick Inference Example

Run the following snippet to test the model with a single image:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("TrafficSignModel_best.keras")

# Load and preprocess image
img = image.load_img("test_image.jpg", target_size=(32, 32))
img = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

# Predict
pred = model.predict(img)
print("Predicted class:", pred.argmax())
```

---

## 📝 Note

This README corresponds to the current commit described as:
> "feat: Initial Model Test Results and Documentation Commit."

Future commits will include additional improvements, extended documentation, and optional real-time demo integration.

