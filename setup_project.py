"""
Setup script for Face Emotion Detection project.
This script helps set up the project environment and checks dependencies.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.7 or higher."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'tensorflow',
        'keras',
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                __import__('cv2')
            elif package == 'Pillow':
                __import__('PIL')
            else:
                __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[X] {package} is NOT installed")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install dependencies from requirements.txt."""
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found")
        return False
    
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("\n[OK] All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n[X] Error installing dependencies")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = ['data', 'models', 'scripts', 'utils']
    required_files = ['requirements.txt']
    
    print("\nChecking project structure...")
    
    all_good = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"[OK] Directory '{dir_name}' exists")
        else:
            print(f"[X] Directory '{dir_name}' is missing")
            all_good = False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"[OK] File '{file_name}' exists")
        else:
            print(f"[X] File '{file_name}' is missing")
            all_good = False
    
    return all_good

def main():
    """Main setup function."""
    print("="*70)
    print("FACE EMOTION DETECTION - PROJECT SETUP")
    print("="*70)
    
    # Check Python version
    print("\n[1] Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Check project structure
    print("\n[2] Checking project structure...")
    if not check_project_structure():
        print("\nWarning: Some project structure elements are missing")
    
    # Check dependencies
    print("\n[3] Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        response = input("\nWould you like to install missing packages? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("\nPlease install missing packages manually:")
            print("  pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies are installed!")
    
    # Check data directory
    print("\n[4] Checking dataset...")
    if os.path.exists('data'):
        emotion_folders = [f for f in os.listdir('data') 
                          if os.path.isdir(os.path.join('data', f))]
        if emotion_folders:
            print(f"[OK] Found {len(emotion_folders)} emotion folders: {emotion_folders}")
        else:
            print("[X] No emotion folders found in 'data' directory")
    else:
        print("[X] 'data' directory not found")
    
    print("\n" + "="*70)
    print("SETUP CHECK COMPLETED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Ensure your dataset is in the 'data' folder")
    print("  2. Run: python scripts/train_model.py")
    print("  3. After training, run: python scripts/evaluate_model.py")
    print("  4. Test with: python scripts/predict_image.py <image_path>")
    print("  5. Real-time: python scripts/webcam_detection.py")

if __name__ == "__main__":
    main()
