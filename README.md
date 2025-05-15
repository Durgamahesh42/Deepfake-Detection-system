
# 🧠 Deepfake Detection System

This is a complete Deepfake Detection System built using a GAN-based attention-enhanced discriminator. It detects whether an uploaded image is a real human face or a GAN-generated deepfake. The system includes a training pipeline, batch/single prediction script, and a Flask web interface.

---

## 📁 Project Structure

```
deepfake_detection/
├── app.py                  # Flask web app for image prediction
├── predict.py              # CLI tool for single/batch prediction
├── train.py                # Model training script with GAN discriminator
├── final_deepfake_detector.h5  # Saved trained model
├── data/
│   ├── uploads/            # Uploaded images from web interface
│   └── real_and_fake_face/ # Dataset directory (real/fake faces)
├── checkpoints/            # Model checkpoints
├── results/                # Training metrics and plots
└── templates/
    └── index.html          # Web UI HTML template
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

### 2. Install Requirements

Ensure Python 3.8+ is installed.

```bash
pip install tensorflow flask numpy opencv-python pillow matplotlib scikit-learn
```

---

## ⚙️ Training the Model

Make sure your dataset is structured like this:
```
data/real_and_fake_face/
├── training_real/
├── training_fake/
```

Run the training script:

```bash
python train.py
```

The model will be saved as `deepfake_detection/final_deepfake_detector.h5`.

---

## 🧪 Prediction Methods

### A. Web Interface

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`. Upload an image to classify it as **REAL** or **FAKE**.

### B. Command-Line Interface

#### Predict a Single Image:

```bash
python predict.py --model path/to/model.h5 --image path/to/image.jpg
```

#### Predict a Folder of Images:

```bash
python predict.py --model path/to/model.h5 --dir path/to/images/ --output results.csv
```

---

## 🔍 Model Architecture

- Custom CNN-based GAN Discriminator
- 5 Convolution Blocks with:
  - LeakyReLU Activation
  - Batch Normalization
  - Dropout
  - MaxPooling
- Attention blocks in mid-layers (3rd and 4th)
- Final Dense layer with Sigmoid for binary classification

---

## 📊 Training & Evaluation

- Training Accuracy: ~89%
- Confidence scores included with every prediction
- Results visualized and saved to `results/`

---

## 📈 Future Enhancements

- Add video deepfake detection
- Streamlit-based UI for faster prototyping
- TensorFlow Lite support for mobile deployment
- Browser extension for real-time detection
- Explainability with attention heatmaps

---
  
- Durga Mahesh Muthinti 
*Guided by Dr. Abhijit Dasgupta, SRM University – AP*

---

## 📄 License

For academic and research use only.
