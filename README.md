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

## 🚀 Features

- EfficientNetB0 Transfer Learning
- Real-world field image augmentation
- Automatic Train / Validation / Test split
- Class imbalance handling using class weights
- Fine-tuning support
- Detailed evaluation metrics
- Confusion Matrix visualization
- Bootstrap confidence interval analysis
- FLOPs estimation
- Deployment-ready inference function

---

## 🧠 Model Architecture

### Base Model
- EfficientNetB0 (Pretrained on ImageNet)

### Custom Classification Head
- GlobalAveragePooling2D
- Dense(256, ReLU)
- Dropout(0.5)
- Dense(128, ReLU)
- Dropout(0.3)
- Softmax Output Layer

### Optimizer
- SGD with Momentum + Nesterov

---

## 📂 Dataset Structure

```bash
Real_Field_Tomato_Dataset/
│
├── Early_Blight/
├── Late_Blight/
├── Healthy/
├── Leaf_Mold/
├── Septoria_Leaf_Spot/
└── ...
```

The script automatically creates:

```bash
train/
validation/
test/
```

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Train Split | 70% |
| Validation Split | 15% |
| Test Split | 15% |
| Initial Learning Rate | 0.01 |
| Fine-Tuning LR | 0.001 |
| Optimizer | SGD |
| Random Seed | 454546 |

---

## 🔄 Data Augmentation

The training pipeline applies realistic agricultural augmentations:
- Rotation
- Horizontal Flip
- Zoom
- Brightness Adjustment
- Width & Height Shift
- Shear Transformation

This improves model robustness in real-world field conditions.

---

## ⚖️ Class Balancing

To handle dataset imbalance, the project uses:

```python
compute_class_weight(class_weight='balanced')
```

This ensures fair learning across all disease classes.

---

## 📊 Evaluation Metrics

The project evaluates performance using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Bootstrap Confidence Interval

---

## 📈 Visualizations

Generated outputs include:
- Training Accuracy Graph
- Training Loss Graph
- Confusion Matrix Heatmap

Saved automatically as:

```bash
real_field_tomato_efficientnetb0_training_history.png
real_field_tomato_efficientnetb0_confusion_matrix.png
```

---

## 💾 Model Saving

The trained model is saved as:

```bash
real_field_tomato_efficientnetb0_model.keras
```

---

## 🔍 Inference Example

```python
model = tf.keras.models.load_model(
    'real_field_tomato_efficientnetb0_model.keras'
)

disease, confidence, all_probs = predict_tomato_disease(
    'path/to/tomato_leaf.jpg'
)

print(f"Predicted: {disease}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `.keras` | Trained model |
| `.csv` | Training & evaluation results |
| `.png` | Confusion matrix |
| `.png` | Training history plots |

---

## 🧪 Performance Highlights

- Transfer Learning using EfficientNetB0
- Real field image classification
- Fine-tuned architecture
- Balanced training strategy
- Deployment-ready prediction pipeline

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/tomato-leaf-disease-classification.git
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas psutil
```

### 3️⃣ Run Notebook / Script

Open in Google Colab and run all cells.

---

## 📌 Future Improvements

- Deploy using Streamlit or Flask
- Mobile App Integration
- Real-time camera detection
- Support for additional crop diseases
- Model quantization for edge devices

---

## 👨‍💻 Author

Developed by **Hamza_Ali**

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub!
