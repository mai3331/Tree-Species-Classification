# Tree-Species-Classification

This project demonstrates the classification of tree leaf images using deep learning. Four different architectures were implemented and compared:  

- **VGG-19 (From Scratch)**  
- **ResNet50 (Transfer Learning)**  
- **Inception V1 (Transfer Learning)**  
- **MobileNetV2 (Transfer Learning)**  

The dataset contains images of 32 tree species.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Pre-Modeling](#pre-modeling)
4. [Models](#models)
    - [VGG-19](#vgg-19-from-scratch)
    - [ResNet50](#resnet50-transfer-learning)
    - [Inception V1](#inception-v1-transfer-learning)
    - [MobileNetV2](#mobilenetv2-transfer-learning)
5. [Evaluation & Metrics](#evaluation--metrics)
6. [Results Comparison](#results-comparison)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Images & Visualizations](#images--visualizations)

---

## Introduction
This project focuses on identifying tree species from leaf images. Classification of plant species is important for biodiversity monitoring, agriculture, and environmental research.  

We explore four architectures, train them on a labeled dataset, and evaluate their performance using **accuracy, precision, recall, F1-score, ROC, and AUC**.

---

## Dataset Description
- **Dataset Name:** Flavia Leaf Dataset  
- **Number of classes:** 32 tree species  
- **Number of images:** ~1900 images  
- **Image size:** 224x224 RGB after preprocessing  
- **Source:** [Kaggle / Flavia Dataset](#)  

The dataset was split into **80% training** and **20% testing**. Images were normalized and optionally augmented to improve model generalization.

---

## Pre-Modeling
- **Preprocessing steps:**
  - Resizing to 224x224
  - RGB conversion
  - Normalization (scaling pixel values to [0,1])
  - One-hot encoding labels
- **Train/Test split:** 80/20
- **Challenges:** Some classes are imbalanced, requiring data augmentation or weighted loss during training.

---

## Models

### VGG-19 (From Scratch)
- Deep convolutional network with 19 layers
- Pros:
  - High representational power
  - Performs well on large datasets
- Cons:
  - Prone to overfitting on small datasets
  - Very large number of parameters, slower training

---

### ResNet50 (Transfer Learning)
- Residual Network with skip connections
- Pros:
  - Avoids vanishing gradients
  - Pre-trained on ImageNet for strong feature extraction
  - Fast convergence with transfer learning
- Cons:
  - Still large and computationally intensive

---

### Inception V1 (Transfer Learning)
- Inception network with parallel convolution filters
- Pros:
  - Efficient with fewer parameters
  - Captures multi-scale features
- Cons:
  - Complex architecture, more tuning required
  - Original weights limited in modern frameworks (used Inception V3 as approximation)

---

### MobileNetV2 (Transfer Learning)
- Lightweight CNN optimized for mobile and edge devices
- Pros:
  - Fast and efficient
  - Smaller memory footprint
- Cons:
  - Slightly lower accuracy on complex datasets
  - May underperform on very fine-grained classification

---

## Evaluation & Metrics
For each model:
- **Confusion Matrix**
- **Classification Report:** precision, recall, F1-score
- **ROC Curve & AUC**
- **Accuracy & Loss plots**

```python
# Example placeholder for metrics
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.show()
