# 🍅 Real Field Tomato Leaf Disease Classification using EfficientNetB0

## 📌 Overview
This project presents a deep learning-based tomato leaf disease classification system trained on real field tomato leaf images using EfficientNetB0 and TensorFlow/Keras.

The model is designed to classify multiple tomato leaf diseases under realistic agricultural conditions with:
- Transfer Learning
- Class Balancing
- Data Augmentation
- Fine-Tuning
- Performance Evaluation Metrics
- Bootstrap Confidence Intervals

The project was developed and trained in Google Colab with GPU acceleration.

---

# 🚀 Features

- ✅ EfficientNetB0 Transfer Learning
- ✅ Real-world field image augmentation
- ✅ Automatic Train / Validation / Test split
- ✅ Class imbalance handling using class weights
- ✅ Fine-tuning support
- ✅ Detailed evaluation metrics
- ✅ Confusion Matrix visualization
- ✅ Bootstrap confidence interval analysis
- ✅ FLOPs estimation
- ✅ Deployment-ready inference function

---

# 🧠 Model Architecture

## Base Model
- EfficientNetB0 (Pretrained on ImageNet)

## Custom Classification Head
- GlobalAveragePooling2D
- Dense(256, ReLU)
- Dropout(0.5)
- Dense(128, ReLU)
- Dropout(0.3)
- Softmax Output Layer

## Optimizer
- SGD with Momentum + Nesterov

---

# 📂 Dataset Structure

```bash
Real_Field_Tomato_Dataset/
│
├── Early_Blight/
├── Late_Blight/
├── Healthy/
├── Leaf_Mold/
├── Septoria_Leaf_Spot/
└── ...
