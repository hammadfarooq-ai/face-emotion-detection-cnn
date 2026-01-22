"""
Face Emotion Detection - Single Image Prediction Script
=======================================================
This script predicts emotions from a single image file.

Usage:
    python scripts/predict_image.py <image_path>
    
Example:
    python scripts/predict_image.py test_image.jpg
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import pickle
import argparse

# Configuration
MODEL_PATH = "models/emotion_model.h5"
IMG_SIZE = 48

# ============================================================================
# FACE DETECTION FUNCTION
# ============================================================================

def detect_face(image):
    """
    Detect face in image using Haar Cascade.
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        face_roi: Cropped face region, or None if no face detected
        face_coords: (x, y, w, h) coordinates of detected face
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade classifier for face detection
    # Try to find the cascade file (usually in OpenCV installation)
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_default.xml',
        'utils/haarcascade_frontalface_default.xml'
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if os.path.exists(path):
            face_cascade = cv2.CascadeClassifier(path)
            break
    
    if face_cascade is None:
        print("Warning: Could not find Haar Cascade file. Using full image.")
        return image, (0, 0, image.shape[1], image.shape[0])
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print("Warning: No face detected. Using full image.")
        return image, (0, 0, image.shape[1], image.shape[0])
    
    # Use the largest face if multiple faces detected
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Extract face region
    face_roi = image[y:y+h, x:x+w]
    
    return face_roi, (x, y, w, h)


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image(image_path):
    """
    Load and preprocess image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        processed_image: Preprocessed image array ready for model
        original_image: Original image for display
        face_coords: Face coordinates if detected
    """
    # Read image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Detect face
    face_roi, face_coords = detect_face(original_image)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    normalized = resized.astype('float32') / 255.0
    
    # Add batch dimension: (1, 48, 48, 3)
    processed = np.expand_dims(normalized, axis=0)
    
    return processed, original_image, face_coords


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion(model, image_array, class_names):
    """
    Predict emotion from preprocessed image.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        class_names: List of emotion class names
        
    Returns:
        predicted_class: Predicted emotion class name
        confidence: Confidence score (probability)
        all_probs: All class probabilities
    """
    # Get predictions
    predictions = model.predict(image_array, verbose=0)
    
    # Get predicted class index
    predicted_idx = np.argmax(predictions[0])
    
    # Get confidence (probability)
    confidence = predictions[0][predicted_idx]
    
    # Get predicted class name
    predicted_class = class_names[predicted_idx]
    
    # Get all probabilities
    all_probs = {class_names[i]: float(predictions[0][i]) 
                 for i in range(len(class_names))}
    
    return predicted_class, confidence, all_probs


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_prediction(image, face_coords, predicted_class, confidence, all_probs):
    """
    Draw bounding box and labels on image.
    
    Args:
        image: Original image (BGR format)
        face_coords: (x, y, w, h) face coordinates
        predicted_class: Predicted emotion
        confidence: Confidence score
        all_probs: All class probabilities
    """
    # Create a copy for drawing
    output_image = image.copy()
    
    # Draw face bounding box
    x, y, w, h = face_coords
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare text
    label = f"{predicted_class}: {confidence:.2%}"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        output_image,
        (x, y - text_height - 10),
        (x + text_width, y),
        (0, 255, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        output_image,
        label,
        (x, y - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )
    
    return output_image


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict emotion from a single image'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the output image (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("FACE EMOTION DETECTION - SINGLE IMAGE PREDICTION")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file not found at {MODEL_PATH}")
        print("Please run 'python scripts/train_model.py' first to train the model.")
        sys.exit(1)
    
    # Load model
    print("\n[Step 1] Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    
    # Load class names
    print("\n[Step 2] Loading class names...")
    with open('models/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    print(f"Classes: {class_names}")
    
    # Preprocess image
    print(f"\n[Step 3] Processing image: {args.image_path}")
    try:
        processed_image, original_image, face_coords = preprocess_image(args.image_path)
        print("Image preprocessed successfully")
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)
    
    # Predict emotion
    print("\n[Step 4] Predicting emotion...")
    predicted_class, confidence, all_probs = predict_emotion(
        model, processed_image, class_names
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Predicted Emotion: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nAll Probabilities:")
    for emotion, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 30)
        bar = '█' * bar_length
        print(f"  {emotion:15s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")
    
    # Visualize and save
    output_image = visualize_prediction(
        original_image, face_coords, predicted_class, confidence, all_probs
    )
    
    # Save output image if requested
    if args.save:
        cv2.imwrite(args.save, output_image)
        print(f"\nOutput image saved to: {args.save}")
    else:
        # Display image (will open in default image viewer)
        output_path = 'prediction_output.jpg'
        cv2.imwrite(output_path, output_image)
        print(f"\nOutput image saved to: {output_path}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETED!")
    print("="*70)
