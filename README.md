# 🔢 Handwritten Digit Classifier (PyTorch + Streamlit)

This project implements a **handwritten digit classification system** using **PyTorch** for model training and **Streamlit** for deployment.  
It classifies digits **0–9** from the **MNIST dataset**, and users can test it via a **web app** that supports **image upload** and **webcam input**.  

---

## 📂 Project Overview
- **Dataset**: MNIST (60,000 training images + 10,000 test images, 28×28 grayscale digits).  
- **Model**: Convolutional Neural Network (CNN) trained from scratch.  
- **Frameworks**: PyTorch for model training, Streamlit for web app interface.  
- **Output**: Predicted digit + class probabilities bar chart.  

---

## 📚 Libraries Used
The project relies on the following libraries:

| Library       | Purpose |
|---------------|---------|
| `torch`       | Deep learning framework (model, training, evaluation). |
| `torchvision` | Access to MNIST dataset, transforms for preprocessing. |
| `numpy`       | Numerical operations. |
| `pillow`      | Image processing. |
| `streamlit`   | Web app deployment. |
| `opencv-python` | Webcam capture support. |


