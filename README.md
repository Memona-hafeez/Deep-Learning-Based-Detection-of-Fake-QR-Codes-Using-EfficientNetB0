# 📌 Deep Learning Based Detection of Fake QR Codes Using EfficientNetB0

This repository contains the implementation of a deep learning approach to detect **fake / counterfeit QR codes** using transfer learning with the **EfficientNetB0** model.

With the rapid rise of QR codes in digital transactions, product authentication, and contactless systems, fake QR codes have become a serious security vulnerability. This project aims to detect fake QR codes by analyzing the image patterns *before decoding* to prevent scams, phishing, and fraud. :contentReference[oaicite:1]{index=1}

---

## 🔍 Project Overview

Most QR code scanners simply decode the QR content (like URLs or text) and do not verify whether the code itself is genuine. This approach introduces vulnerabilities where attackers can replace authentic codes with malicious ones.

To mitigate this, we use a **deep learning classification model** that:
- Takes QR code images as input
- Learns visual patterns showing real vs fake codes 
- Classifies them accurately with high performance

This provides a **pre-decode visual authentication layer** for QR code security. :contentReference[oaicite:2]{index=2}

---

## 🚀 Features

✔ Binary classification of QR codes (Real vs Fake)  
✔ Uses EfficientNetB0 + transfer learning  
✔ End-to-end training pipeline  
✔ Performance reporting (accuracy, precision, recall)  
✔ Easy to extend for mobile deployment  
✔ Suitable for integration with QR scanning apps :contentReference[oaicite:3]{index=3}

---

## 🧠 How It Works

1. **Dataset Loading**: Load labeled images of real and fake QR codes  
2. **Preprocessing**: Resize, normalize, and prepare images for EfficientNetB0  
3. **Model Architecture**:
   - Backbone: EfficientNetB0 pretrained on ImageNet  
   - Custom classification head for binary output  
   - Softmax or Sigmoid for two-class prediction
4. **Training**: Train with binary cross-entropy loss + optimizer like Adam  
5. **Evaluation**: Report accuracy, confusion matrix, precision, recall  
6. **Inference**: Predict label for new QR code image inputs :contentReference[oaicite:4]{index=4}

---

## 🛠 Project Structure

```
Deep-Learning-Based-Detection-of-Fake-QR-Codes-Using-EfficientNetB0/
├── data/                     # QR code image dataset
│   ├── real/                # Genuine QR codes
│   └── fake/                # Fake QR codes
├── models/                  # Saved model weights
├── notebooks/               # Jupyter notebooks (training, EDA)
├── src/
│   ├── data_loader.py       # Dataset split & preprocessing
│   ├── model.py             # EfficientNetB0 model definition
│   ├── train.py             # Train script
│   ├── evaluate.py          # Evaluation script
│   └── predict.py           # Inference script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE
```

---

## ⚙️ Dependencies

Make sure you have the following installed:

```
Python 3.8+
tensorflow
keras
numpy
opencv-python
matplotlib
scikit-learn
efficientnet
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📦 Training the Model

1. Prepare dataset directory with `real/` and `fake/` subfolders  
2. Run training:

```bash
python src/train.py \
    --data_dir data/ \
    --epochs 25 \
    --batch_size 32 \
    --save_model models/efficient_qr_detector.h5
```

---

## 📊 Evaluation

After training:

```bash
python src/evaluate.py \
    --model models/efficient_qr_detector.h5 \
    --data_dir data/
```

This prints:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  

---

## 🚀 Inference

To test a new image:

```bash
python src/predict.py \
    --model models/efficient_qr_detector.h5 \
    --image sample_qr.png
```

Output:

```
Prediction: FAKE (confidence: 98.7%)
```

---

## 🧪 Dataset Notes

✔ Dataset contains images labeled Real vs Fake  
✔ Images are standardized to same size  
✔ Data augmentation allowed to increase robustness  
✔ Suggested input size: 224 × 224 :contentReference[oaicite:5]{index=5}

---

## 📈 Results (Example)

After training:

| Metric       | Value |
|--------------|-------|
| Accuracy     | 99.98% |
| Precision    | 99.9% |
| Recall       | 99.9% |

Exceptional performance indicates strong visual discrimination with EfficientNetB0. :contentReference[oaicite:6]{index=6}

---

## 🎯 Applications

✔ Mobile QR code scanner with authenticity check  
✔ Point-of-sale QR security  
✔ Anti-phishing systems  
✔ Product packaging verification  
✔ IoT security checkpoints

---

## 🛠 Future Improvements

✨ Add URL content scanning  
✨ Integrate with smartphone apps  
✨ Adversarial robustness testing  
✨ Explainable AI visualization of suspicious patterns

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ **If you find this work useful, please give it a star!**
