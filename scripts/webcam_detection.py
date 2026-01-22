"""
Face Emotion Detection - Real-time Webcam Detection Script
==========================================================
This script performs real-time emotion detection using webcam.

Usage:
    python scripts/webcam_detection.py
    
Press 'q' to quit the application.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

# Configuration
MODEL_PATH = "models/emotion_model.h5"
IMG_SIZE = 48

# ============================================================================
# FACE DETECTION SETUP
# ============================================================================

def load_face_cascade():
    """Load Haar Cascade classifier for face detection."""
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_default.xml',
        'utils/haarcascade_frontalface_default.xml'
    ]
    
    for path in cascade_paths:
        if os.path.exists(path):
            return cv2.CascadeClassifier(path)
    
    raise FileNotFoundError(
        "Could not find Haar Cascade file. "
        "Please ensure OpenCV is properly installed."
    )


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_face(face_roi):
    """
    Preprocess face ROI for model prediction.
    
    Args:
        face_roi: Face region from webcam (BGR format)
        
    Returns:
        processed: Preprocessed image array ready for model
    """
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    normalized = resized.astype('float32') / 255.0
    
    # Add batch dimension
    processed = np.expand_dims(normalized, axis=0)
    
    return processed


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion(model, face_array, class_names):
    """
    Predict emotion from face image.
    
    Args:
        model: Trained Keras model
        face_array: Preprocessed face array
        class_names: List of emotion class names
        
    Returns:
        predicted_class: Predicted emotion
        confidence: Confidence score
    """
    # Get prediction
    predictions = model.predict(face_array, verbose=0)
    
    # Get predicted class
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    return predicted_class, confidence


# ============================================================================
# DRAWING FUNCTIONS
# ============================================================================

def draw_emotion_label(frame, x, y, w, h, emotion, confidence):
    """
    Draw bounding box and emotion label on frame.
    
    Args:
        frame: Video frame
        x, y, w, h: Face bounding box coordinates
        emotion: Predicted emotion
        confidence: Confidence score
    """
    # Draw face bounding box
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    label = f"{emotion}: {confidence:.1%}"
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width + 10, y),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x + 5, y - 5),
        font,
        font_scale,
        (0, 0, 0),  # Black text
        text_thickness
    )


def draw_info_panel(frame, fps, class_names):
    """
    Draw information panel on frame.
    
    Args:
        frame: Video frame
        fps: Current FPS
        class_names: List of emotion classes
    """
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    y_offset = 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                font, font_scale, color, thickness)
    y_offset += 20
    cv2.putText(frame, "Press 'q' to quit", (20, y_offset), 
                font, font_scale, color, thickness)
    y_offset += 20
    cv2.putText(frame, f"Classes: {len(class_names)}", (20, y_offset), 
                font, font_scale, color, thickness)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FACE EMOTION DETECTION - REAL-TIME WEBCAM")
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
    
    # Load class names
    print("\n[Step 2] Loading class names...")
    with open('models/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    print(f"Classes: {class_names}")
    
    # Load face cascade
    print("\n[Step 3] Loading face detection cascade...")
    face_cascade = load_face_cascade()
    print("Face cascade loaded successfully")
    
    # Initialize webcam
    print("\n[Step 4] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nWebcam initialized successfully")
    print("\n" + "="*70)
    print("Starting real-time emotion detection...")
    print("Press 'q' to quit")
    print("="*70 + "\n")
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                emotion, confidence = predict_emotion(model, processed_face, class_names)
                
                # Draw bounding box and label
                draw_emotion_label(frame, x, y, w, h, emotion, confidence)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                current_fps = 30 / time_diff
                fps_start_time = fps_end_time
            
            # Draw info panel
            draw_info_panel(frame, current_fps, class_names)
            
            # Display frame
            cv2.imshow('Face Emotion Detection', frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Goodbye!")
    
    print("\n" + "="*70)
    print("REAL-TIME DETECTION ENDED")
    print("="*70)
