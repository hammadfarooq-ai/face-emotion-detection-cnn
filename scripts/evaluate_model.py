"""
Face Emotion Detection - Evaluation Script
==========================================
This script evaluates the trained model on the test set and generates:
- Accuracy and Loss metrics
- Confusion Matrix
- Classification Report (precision, recall, F1-score)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import cv2

# Configuration
DATA_DIR = "data"
MODEL_PATH = "models/emotion_model.h5"
IMG_SIZE = 48
BATCH_SIZE = 32

# ============================================================================
# LOAD DATA (same as training script)
# ============================================================================

def load_images_from_folders(data_dir):
    """Load images from emotion-labeled folders."""
    images = []
    labels = []
    
    emotion_folders = [folder for folder in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, folder))]
    emotion_folders.sort()
    
    for emotion in emotion_folders:
        emotion_path = os.path.join(data_dir, emotion)
        image_files = [f for f in os.listdir(emotion_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(emotion)
            except Exception as e:
                continue
    
    return np.array(images), np.array(labels)


def prepare_test_data(images, labels, label_encoder):
    """Prepare test data with same preprocessing as training."""
    labels_encoded = label_encoder.transform(labels)
    labels_onehot = keras.utils.to_categorical(labels_encoded, len(label_encoder.classes_))
    return images, labels_onehot, labels_encoded


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels (integer encoded)
        y_pred: Predicted labels (integer encoded)
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Emotion', fontsize=12)
    axes[0].set_ylabel('True Emotion', fontsize=12)
    
    # Plot 2: Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Emotion', fontsize=12)
    axes[1].set_ylabel('True Emotion', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Save to file
    with open('models/classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("="*70 + "\n")
        f.write(report)
    print("\nClassification report saved to: models/classification_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FACE EMOTION DETECTION - MODEL EVALUATION")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file not found at {MODEL_PATH}")
        print("Please run 'python scripts/train_model.py' first to train the model.")
        exit(1)
    
    # Load model
    print("\n[Step 1] Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    
    # Load label encoder and class names
    print("\n[Step 2] Loading label encoder and class names...")
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('models/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    print(f"Classes: {class_names}")
    
    # Load test data
    print("\n[Step 3] Loading test data...")
    images, labels = load_images_from_folders(DATA_DIR)
    print(f"Total images loaded: {len(images)}")
    
    # Split data (using same split as training - in production, you'd save test indices)
    # For now, we'll use a subset for evaluation
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Re-encode labels to match training
    temp_encoder = LabelEncoder()
    labels_encoded = temp_encoder.fit_transform(labels)
    labels_onehot = keras.utils.to_categorical(labels_encoded, len(temp_encoder.classes_))
    
    # Use 20% as test set (same as training script logic)
    _, X_test, _, y_test = train_test_split(
        images, labels_onehot, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_onehot
    )
    
    # Further split to get actual test set (10% of total)
    _, X_test_final, _, y_test_final = train_test_split(
        X_test, y_test,
        test_size=0.5,  # 50% of 20% = 10% of total
        random_state=42,
        stratify=y_test
    )
    
    print(f"Test set size: {X_test_final.shape[0]} images")
    
    # Evaluate model
    print("\n[Step 4] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_final, y_test_final, verbose=0)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    print("\n[Step 5] Generating predictions...")
    y_pred_probs = model.predict(X_test_final, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_final, axis=1)
    
    # Plot confusion matrix
    print("\n[Step 6] Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names, 'models/confusion_matrix.png')
    
    # Print classification report
    print("\n[Step 7] Generating classification report...")
    print_classification_report(y_true, y_pred, class_names)
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(class_names):
        if cm[i, i] + cm[i].sum() - cm[i, i] > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name:15s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/confusion_matrix.png")
    print("  - models/classification_report.txt")
