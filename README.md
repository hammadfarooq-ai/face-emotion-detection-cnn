# Face Emotion Detection Project

A complete deep learning project for detecting human emotions from facial images using Convolutional Neural Networks (CNN) with TensorFlow and Keras.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a complete end-to-end solution for facial emotion recognition. It can detect 7 different emotions:
- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😲 **Surprise**

The system includes:
- Model training with data augmentation
- Model evaluation with comprehensive metrics
- Single image prediction
- Real-time webcam emotion detection

## ✨ Features

- **Deep Learning Model**: CNN-based architecture optimized for emotion detection
- **Data Preprocessing**: Automatic image resizing, normalization, and augmentation
- **Face Detection**: Automatic face detection using OpenCV Haar Cascade
- **Real-time Detection**: Live emotion detection from webcam feed
- **Comprehensive Evaluation**: Confusion matrix, classification report, and accuracy metrics
- **Production Ready**: Clean code structure with proper error handling

## 📁 Project Structure

```
Facial Emotion Detetion Data/
│
├── data/                          # Dataset folder (emotion-labeled subfolders)
│   ├── angry/                     # Angry emotion images
│   ├── disgust/                   # Disgust emotion images
│   ├── fear/                      # Fear emotion images
│   ├── happy/                     # Happy emotion images
│   ├── neutral/                   # Neutral emotion images
│   ├── sad/                       # Sad emotion images
│   └── surprise/                  # Surprise emotion images
│
├── models/                        # Saved models and artifacts
│   ├── emotion_model.h5          # Trained model (generated after training)
│   ├── label_encoder.pkl         # Label encoder (generated after training)
│   ├── class_names.pkl            # Class names (generated after training)
│   ├── training_history.png       # Training curves (generated after training)
│   ├── confusion_matrix.png       # Confusion matrix (generated after evaluation)
│   └── classification_report.txt  # Classification report (generated after evaluation)
│
├── scripts/                       # Main Python scripts
│   ├── train_model.py            # Training script
│   ├── evaluate_model.py         # Evaluation script
│   ├── predict_image.py          # Single image prediction script
│   └── webcam_detection.py       # Real-time webcam detection script
│
├── utils/                         # Utility functions
│   └── __init__.py               # Utility package init file
│
├── requirements.txt               # Python dependencies
├── setup_project.py              # Project setup and dependency checker
├── PROJECT_STRUCTURE.txt          # Detailed project structure documentation
├── run_training.bat              # Quick training script (Windows)
├── run_evaluation.bat            # Quick evaluation script (Windows)
├── run_webcam.bat                # Quick webcam script (Windows)
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone or Navigate to Project Directory

```bash
cd "D:\Projects\Facial Emotion Detetion Data"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script to check and install dependencies:

```bash
python setup_project.py
```

### Step 3: Verify Installation

The setup script will check:
- Python version
- Required packages
- Project structure
- Dataset availability

## 📖 Usage

### 1. Train the Model

Train the CNN model on your dataset:

```bash
python scripts/train_model.py
```

**What it does:**
- Loads images from `data/` folders
- Preprocesses images (resize to 48x48, normalize)
- Splits data: 70% train, 20% validation, 10% test
- Builds and trains CNN model
- Saves model to `models/emotion_model.h5`
- Generates training history plot

**Expected Output:**
- Model architecture summary
- Training progress with accuracy/loss metrics
- Saved model file
- Training history visualization

### 2. Evaluate the Model

Evaluate the trained model on test set:

```bash
python scripts/evaluate_model.py
```

**What it does:**
- Loads trained model
- Evaluates on test set
- Generates confusion matrix (saved as image)
- Creates classification report (precision, recall, F1-score)
- Displays per-class accuracy

**Generated Files:**
- `models/confusion_matrix.png` - Visual confusion matrix
- `models/classification_report.txt` - Detailed metrics

### 3. Predict Emotion from Single Image

Predict emotion from a single image file:

```bash
python scripts/predict_image.py <image_path>
```

**Examples:**

```bash
# Basic prediction
python scripts/predict_image.py test_face.jpg

# Save output image
python scripts/predict_image.py test_face.jpg --save output.jpg
```

**Features:**
- Automatic face detection
- Emotion prediction with confidence scores
- Visual output with bounding box and label
- Shows all emotion probabilities

### 4. Real-time Webcam Detection

Detect emotions in real-time from webcam:

```bash
python scripts/webcam_detection.py
```

**Controls:**
- Press `q` to quit
- FPS counter displayed
- Real-time face detection and emotion prediction

**Features:**
- Live face detection
- Real-time emotion prediction
- Visual feedback with bounding boxes
- Performance metrics (FPS)

## 🏗️ Model Architecture

The CNN model consists of:

```
Input Layer: 48x48x3 (RGB images)

Convolutional Block 1:
  - Conv2D(32 filters, 3x3) + ReLU
  - Conv2D(32 filters, 3x3) + ReLU
  - MaxPooling2D(2x2)
  - Dropout(0.25)

Convolutional Block 2:
  - Conv2D(64 filters, 3x3) + ReLU
  - Conv2D(64 filters, 3x3) + ReLU
  - MaxPooling2D(2x2)
  - Dropout(0.25)

Convolutional Block 3:
  - Conv2D(128 filters, 3x3) + ReLU
  - Conv2D(128 filters, 3x3) + ReLU
  - MaxPooling2D(2x2)
  - Dropout(0.25)

Fully Connected Layers:
  - Flatten
  - Dense(512) + ReLU
  - Dropout(0.5)
  - Dense(256) + ReLU
  - Dropout(0.5)
  - Dense(7) + Softmax (Output: 7 emotion classes)
```

**Training Configuration:**
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Data Augmentation**: Rotation, shifts, flips, zoom

## 📊 Dataset

The dataset should be organized in the following structure:

```
data/
├── angry/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── disgust/
│   └── ...
├── fear/
│   └── ...
├── happy/
│   └── ...
├── neutral/
│   └── ...
├── sad/
│   └── ...
└── surprise/
    └── ...
```

**Current Dataset Statistics:**
- Angry: 5,920 images
- Disgust: 5,920 images
- Fear: 5,920 images
- Happy: 11,398 images
- Neutral: 8,166 images
- Sad: 6,535 images
- Surprise: 5,920 images
- **Total**: ~49,779 images

**Data Split:**
- Training: ~70%
- Validation: ~20%
- Test: ~10%

## 📈 Results

After training, you can expect:

- **Model Performance**: Accuracy metrics on test set
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Per-class precision, recall, and F1-scores
- **Training Curves**: Accuracy and loss plots over epochs

Results are saved in the `models/` directory after training and evaluation.

## 🔧 Technical Details

### Image Preprocessing

1. **Face Detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
2. **Resizing**: Images resized to 48x48 pixels
3. **Normalization**: Pixel values normalized to [0, 1] range
4. **Color Conversion**: BGR to RGB conversion for proper display

### Data Augmentation

Applied during training:
- Rotation: ±10 degrees
- Width/Height shifts: ±10%
- Horizontal flip: Enabled
- Zoom: ±10%

### Face Detection

- **Method**: Haar Cascade Classifier
- **Scale Factor**: 1.1
- **Min Neighbors**: 5
- **Min Size**: 30x30 pixels

### Model Saving

- **Format**: H5 (`.h5` or `.keras`)
- **Includes**: Model architecture, weights, optimizer state
- **Additional Files**: Label encoder, class names (pickle format)

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: Model file not found at models/emotion_model.h5
```
**Solution**: Run `python scripts/train_model.py` first to train the model.

**2. OpenCV Haar Cascade not found**
```
Warning: Could not find Haar Cascade file
```
**Solution**: OpenCV should include this file. If missing, download from OpenCV repository or use full image mode.

**3. Out of Memory Error**
```
RuntimeError: Out of memory
```
**Solution**: 
- Reduce `BATCH_SIZE` in training script
- Reduce `IMG_SIZE` (e.g., from 48 to 32)
- Use fewer training images

**4. Webcam not opening**
```
Error: Could not open webcam
```
**Solution**:
- Check if webcam is connected
- Close other applications using the webcam
- Try different camera index: `cv2.VideoCapture(1)`

**5. Import errors**
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Performance Tips

- **Faster Training**: Use GPU if available (TensorFlow will automatically detect)
- **Better Accuracy**: Train for more epochs or use larger model
- **Real-time Performance**: Reduce image size or use lighter model architecture

## 📝 Requirements

See `requirements.txt` for complete list. Main dependencies:

- `tensorflow>=2.15.0` - Deep learning framework
- `keras>=2.15.0` - High-level neural networks API
- `opencv-python>=4.8.0` - Computer vision library
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Plotting library
- `scikit-learn>=1.3.0` - Machine learning utilities
- `Pillow>=10.0.0` - Image processing

## 🎓 Learning Resources

This project is designed for Data Science students. Key concepts demonstrated:

- **Deep Learning**: CNN architecture for image classification
- **Data Preprocessing**: Image normalization and augmentation
- **Model Training**: Hyperparameter tuning and callbacks
- **Model Evaluation**: Metrics and visualization
- **Computer Vision**: Face detection and image processing
- **Production Deployment**: Real-time inference pipeline

## 📄 License

This project is provided as-is for educational purposes.

## 👤 Author

Created as a complete Face Emotion Detection project for educational use.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- OpenCV for computer vision tools
- Dataset contributors for emotion-labeled images

---

**Happy Coding! 🚀**

For questions or issues, refer to the code comments or `PROJECT_STRUCTURE.txt` for detailed documentation.
