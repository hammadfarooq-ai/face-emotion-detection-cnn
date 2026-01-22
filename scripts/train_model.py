"""
Face Emotion Detection - Training Script
========================================
This script trains a CNN model to detect emotions from facial images.

Steps:
1. Load images from dataset folders
2. Preprocess images (resize, normalize)
3. Split data into train/validation/test sets
4. Build CNN model architecture
5. Train the model with callbacks
6. Save the trained model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = "data"  # Root directory containing emotion folders
IMG_SIZE = 48  # Image size (48x48 is common for emotion detection)
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "models/emotion_model.h5"
VALIDATION_SPLIT = 0.2  # 20% for validation
TEST_SPLIT = 0.1  # 10% for test (from remaining 80%)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ============================================================================

def load_images_from_folders(data_dir):
    """
    Load images from emotion-labeled folders.
    
    Args:
        data_dir: Path to the data directory containing emotion folders
        
    Returns:
        images: List of preprocessed image arrays
        labels: List of emotion labels (folder names)
    """
    images = []
    labels = []
    
    # Get all emotion folders (each folder name is an emotion label)
    emotion_folders = [folder for folder in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, folder))]
    emotion_folders.sort()  # Sort for consistent label ordering
    
    print(f"Found {len(emotion_folders)} emotion classes: {emotion_folders}")
    
    # Load images from each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(data_dir, emotion)
        image_files = [f for f in os.listdir(emotion_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Loading {len(image_files)} images from '{emotion}' folder...")
        
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)
            
            try:
                # Read image using OpenCV (handles BGR format)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert BGR to RGB (OpenCV reads as BGR, but we need RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image to fixed size
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize pixel values to [0, 1] range
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(emotion)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    return np.array(images), np.array(labels)


# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================

def prepare_data(images, labels):
    """
    Split data into train/validation/test sets and encode labels.
    
    Args:
        images: Array of image arrays
        labels: Array of emotion labels
        
    Returns:
        X_train, X_val, X_test: Image arrays for train/val/test
        y_train, y_val, y_test: One-hot encoded labels
        label_encoder: LabelEncoder object for later use
    """
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Get class names in order
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    
    print(f"\nClass mapping:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")
    print(f"\nTotal classes: {num_classes}")
    
    # Convert to one-hot encoding (required for categorical crossentropy)
    labels_onehot = keras.utils.to_categorical(labels_encoded, num_classes)
    
    # First split: separate test set (10% of total)
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels_onehot, 
        test_size=TEST_SPLIT, 
        random_state=42, 
        stratify=labels_onehot  # Maintain class distribution
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust validation split to account for test set already removed
    val_size_adjusted = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Validation set: {X_val.shape[0]} images")
    print(f"  Test set: {X_test.shape[0]} images")
    print(f"  Image shape: {X_train.shape[1:]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_names


# ============================================================================
# STEP 3: BUILD CNN MODEL
# ============================================================================

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model for emotion classification.
    
    Architecture:
    - Convolutional layers with ReLU activation
    - MaxPooling layers for downsampling
    - Dropout layers to prevent overfitting
    - Dense layers for classification
    - Softmax output for multi-class prediction
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of emotion classes
        
    Returns:
        model: Compiled Keras model
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),  # Drop 25% of neurons to prevent overfitting
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten the 3D feature maps to 1D feature vector
        layers.Flatten(),
        
        # Fully Connected (Dense) Layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Higher dropout in dense layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer: Softmax for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ============================================================================
# STEP 4: COMPILE AND TRAIN MODEL
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Compile and train the model with callbacks.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        
    Returns:
        history: Training history object
    """
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']
    )
    
    # Print model architecture
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    # Define callbacks
    callbacks = [
        # Early stopping: stop training if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Wait 10 epochs before stopping
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint: save the best model during training
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Data augmentation for training (helps model generalize better)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train)
    
    # Train the model
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ============================================================================
# STEP 5: PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\nTraining history plot saved to 'models/training_history.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*50)
    print("FACE EMOTION DETECTION - TRAINING")
    print("="*50)
    
    # Step 1: Load images
    print("\n[Step 1] Loading images from dataset...")
    images, labels = load_images_from_folders(DATA_DIR)
    print(f"Total images loaded: {len(images)}")
    
    # Step 2: Prepare data
    print("\n[Step 2] Preparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_names = prepare_data(images, labels)
    
    # Save label encoder and class names for later use
    import pickle
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    print("\nLabel encoder and class names saved to 'models/' directory")
    
    # Step 3: Build model
    print("\n[Step 3] Building CNN model...")
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # 3 channels for RGB
    num_classes = len(class_names)
    model = build_cnn_model(input_shape, num_classes)
    
    # Step 4: Train model
    print("\n[Step 4] Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 5: Plot training history
    print("\n[Step 5] Plotting training history...")
    plot_training_history(history)
    
    # Final evaluation on test set
    print("\n[Step 6] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model (best weights should already be saved by ModelCheckpoint)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"  1. Run 'python scripts/evaluate_model.py' for detailed evaluation")
    print(f"  2. Run 'python scripts/predict_image.py <image_path>' for single image prediction")
    print(f"  3. Run 'python scripts/webcam_detection.py' for real-time detection")
