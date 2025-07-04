# 🐶 Cats vs Dogs Image Classification

This project implements a simple Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images as either cats or dogs. The model was trained on a subset of the Kaggle Dogs vs Cats dataset.

---

## 📁 Dataset Used

- **Source**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Structure**:
- data/
- ├── cat/
- └── dog/

  
Each folder contains corresponding image samples of cats and dogs.

---

## ⚙️ Preprocessing

- All images resized to **150×150**
- Pixel values normalized to [0, 1] using `rescale=1./255`
- Used `ImageDataGenerator` to perform an **80/20 training-validation split**

---

## 🧠 Model Architecture

A simple CNN with 3 convolutional layers:

- `Conv2D(32) → MaxPooling2D`
- `Conv2D(64) → MaxPooling2D`
- `Conv2D(128) → MaxPooling2D`
- `Flatten → Dropout(0.5) → Dense(512) → Dense(1, activation='sigmoid')`

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Final Activation**: Sigmoid (for binary classification)

---

## ✅ Validation Accuracy

Achieved **88.4%** accuracy on the validation set after 5 epochs.  
---
