import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pytesseract
from PIL import Image
import gdown
import zipfile
import json
import pickle
import random
from pathlib import Path
import albumentations as A
import re

# Configuration
config = {
    "kaggle_dataset_url": "https://www.kaggle.com/datasets/nikhilroxtomar/handwritten-prescription-dataset/download",
    "local_dataset_path": "prescription_dataset",
    "image_size": (512, 512),
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "test_split": 0.1,
    "model_save_path": "models/prescription_detector.h5",
    "label_encoder_path": "models/label_encoder.pkl",
    "class_names": ["background", "hospital", "patient", "date", "medicine", "dosage", "usage", "doctor"]
}

def download_and_extract_dataset():
    """
    Download the prescription dataset from Kaggle
    Note: Requires Kaggle API credentials in ~/.kaggle/kaggle.json
    """
    # Create directory if it doesn't exist
    os.makedirs(config["local_dataset_path"], exist_ok=True)
    
    # Check if dataset already exists
    if len(os.listdir(config["local_dataset_path"])) > 0:
        print(f"Dataset already exists at {config['local_dataset_path']}")
        return
    
    try:
        import kaggle
        # Download the dataset
        kaggle.api.dataset_download_files(
            "nikhilroxtomar/handwritten-prescription-dataset",
            path=config["local_dataset_path"],
            unzip=True
        )
        print(f"Dataset downloaded and extracted to {config['local_dataset_path']}")
    except:
        print("Error downloading from Kaggle API. Please ensure you have set up Kaggle API credentials.")
        print("Attempting alternative download method...")
        try:
            # Alternative: Download a sample dataset with gdown (Google Drive)
            url = "https://drive.google.com/uc?id=SAMPLE_DATASET_ID"  # Replace with actual shared dataset ID
            output = os.path.join(config["local_dataset_path"], "dataset.zip")
            gdown.download(url, output, quiet=False)
            
            # Extract the zip file
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(config["local_dataset_path"])
            
            # Remove the zip file
            os.remove(output)
            print(f"Dataset downloaded and extracted to {config['local_dataset_path']}")
        except:
            print("Failed to download dataset. Please download manually from Kaggle and place in the configured directory.")
            print(f"Dataset path: {config['local_dataset_path']}")

def load_dataset():
    """
    Load and prepare dataset for training
    """
    images = []
    labels = []
    annotations = []
    
    # Check first if dataset exists
    if not os.path.exists(config["local_dataset_path"]):
        print(f"Dataset not found at {config['local_dataset_path']}. Please download first.")
        return None, None, None
    
    # Look for annotations file (assuming JSON format)
    annotation_files = list(Path(config["local_dataset_path"]).glob("*.json"))
    if not annotation_files:
        print("No annotation files found. Looking for CSV files...")
        annotation_files = list(Path(config["local_dataset_path"]).glob("*.csv"))
    
    if annotation_files:
        # Load annotations
        annotation_file = annotation_files[0]
        if str(annotation_file).endswith('.json'):
            with open(annotation_file, 'r') as f:
                all_annotations = json.load(f)
        elif str(annotation_file).endswith('.csv'):
            all_annotations = pd.read_csv(annotation_file).to_dict('records')
        else:
            print(f"Unsupported annotation format: {annotation_file}")
            return None, None, None
            
        print(f"Loaded annotations from {annotation_file}")
        
        # Process each image and its annotations
        for item in all_annotations:
            if isinstance(item, dict):
                image_path = os.path.join(config["local_dataset_path"], item.get('image_path', ''))
                if not os.path.exists(image_path):
                    # Try alternative paths
                    image_filename = os.path.basename(item.get('image_path', ''))
                    alternative_paths = list(Path(config["local_dataset_path"]).glob(f"**/{image_filename}"))
                    if alternative_paths:
                        image_path = str(alternative_paths[0])
                    else:
                        continue
                        
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, config["image_size"])
                    images.append(img)
                    
                    # Process annotations for this image
                    img_annotations = item.get('annotations', [])
                    if img_annotations:
                        for annot in img_annotations:
                            category = annot.get('category', 'unknown')
                            bbox = annot.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
                            
                            # Map category to class index
                            if category in config["class_names"]:
                                class_idx = config["class_names"].index(category)
                            else:
                                class_idx = 0  # background
                                
                            labels.append(class_idx)
                            annotations.append({
                                'image_path': image_path,
                                'bbox': bbox,
                                'class_idx': class_idx,
                                'class_name': category
                            })
    else:
        # If no annotation file, try to infer from directory structure
        print("No annotation file found. Attempting to load from directory structure...")
        for class_name in config["class_names"]:
            if class_name == "background":
                continue
                
            class_dir = os.path.join(config["local_dataset_path"], class_name)
            if os.path.exists(class_dir) and os.path.isdir(class_dir):
                image_files = list(Path(class_dir).glob("*.jpg")) + list(Path(class_dir).glob("*.png"))
                
                for img_path in image_files:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img = cv2.resize(img, config["image_size"])
                    images.append(img)
                    
                    class_idx = config["class_names"].index(class_name)
                    labels.append(class_idx)
                    
                    # Create a full image annotation since we don't have specific bboxes
                    h, w = img.shape[:2]
                    annotations.append({
                        'image_path': str(img_path),
                        'bbox': [0, 0, w, h],
                        'class_idx': class_idx,
                        'class_name': class_name
                    })
    
    # If still no annotations, create synthetic dataset for demonstration
    if not images or not annotations:
        print("No valid dataset found. Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset()
    
    print(f"Loaded {len(images)} images with {len(annotations)} annotations")
    return np.array(images), np.array(labels), annotations

def create_synthetic_dataset():
    """
    Create a synthetic dataset for demonstration when no real dataset is available
    """
    images = []
    labels = []
    annotations = []
    
    # Create directory for synthetic data
    synthetic_dir = os.path.join(config["local_dataset_path"], "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Define text examples for each class
    text_examples = {
        "hospital": ["City General Hospital", "Medical Center", "Community Health Clinic"],
        "patient": ["Patient: John Doe", "Name: Sarah Smith", "Patient ID: 123456"],
        "date": ["Date: 04/20/2025", "Issued: 01/15/2025", "Prescription Date: March 10, 2025"],
        "medicine": ["Amoxicillin 500mg", "Ibuprofen", "Metformin 1000mg"],
        "dosage": ["Take 1 tablet", "2 pills", "5ml daily"],
        "usage": ["Take with food", "Use as directed", "Take 3 times a day"],
        "doctor": ["Dr. James Wilson", "Prescribed by: Dr. Smith", "Physician: Dr. Johnson"]
    }
    
    # Create synthetic images
    num_synthetic_images = 200
    for i in range(num_synthetic_images):
        # Create blank image
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Generate random annotations
        num_elements = random.randint(3, 7)
        used_classes = random.sample(list(text_examples.keys()), min(num_elements, len(text_examples)))
        
        image_annotations = []
        
        for cls in used_classes:
            # Select random text for this class
            text = random.choice(text_examples[cls])
            
            # Random position and size
            x = random.randint(20, 400)
            y = random.randint(20, 400)
            
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = random.uniform(0.5, 1.0)
            font_thickness = random.randint(1, 2)
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Add some noise/distortion
            if random.random() > 0.7:
                # Add slight rotation
                M = cv2.getRotationMatrix2D((x + text_width//2, y + text_height//2), 
                                           random.uniform(-10, 10), 1)
                img = cv2.warpAffine(img, M, (512, 512))
            
            # Draw text
            cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), font_thickness)
            
            # Create annotation
            class_idx = config["class_names"].index(cls)
            bbox = [x, y - text_height, text_width, text_height + baseline]
            
            # Add to annotations
            image_annotations.append({
                'category': cls,
                'bbox': bbox,
                'class_idx': class_idx
            })
        
        # Save the synthetic image
        image_path = os.path.join(synthetic_dir, f"synthetic_{i}.png")
        cv2.imwrite(image_path, img)
        
        # Add to dataset
        images.append(img)
        
        # For each annotation in the image
        for annot in image_annotations:
            labels.append(annot['class_idx'])
            annotations.append({
                'image_path': image_path,
                'bbox': annot['bbox'],
                'class_idx': annot['class_idx'],
                'class_name': annot['category']
            })
    
    print(f"Created synthetic dataset with {len(images)} images and {len(annotations)} annotations")
    return np.array(images), np.array(labels), annotations

def data_augmentation():
    """
    Create augmentation pipeline for training data
    """
    return A.Compose([
        A.RandomRotate90(p=0.2),
        A.Flip(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def augment_dataset(images, annotations):
    """
    Apply augmentations to dataset
    """
    augmented_images = []
    augmented_annotations = []
    augmented_labels = []
    
    aug = data_augmentation()
    
    # Group annotations by image
    image_to_annotations = {}
    for annot in annotations:
        img_path = annot['image_path']
        if img_path not in image_to_annotations:
            image_to_annotations[img_path] = []
        image_to_annotations[img_path].append(annot)
    
    # Apply augmentations to each image and its annotations
    for i, img in enumerate(images):
        img_path = annotations[i]['image_path']
        img_annotations = image_to_annotations.get(img_path, [])
        
        # Extract bboxes and labels for this image
        bboxes = []
        class_labels = []
        
        for annot in img_annotations:
            bboxes.append(annot['bbox'])
            class_labels.append(annot['class_idx'])
        
        # Apply augmentation
        if bboxes and class_labels:
            augmented = aug(image=img, bboxes=bboxes, class_labels=class_labels)
            augmented_img = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']
            
            # Add augmented image
            augmented_images.append(augmented_img)
            
            # Add augmented annotations
            for j, (bbox, class_idx) in enumerate(zip(augmented_bboxes, augmented_class_labels)):
                augmented_labels.append(class_idx)
                augmented_annotations.append({
                    'image_path': f"augmented_{i}_{j}",
                    'bbox': bbox,
                    'class_idx': class_idx,
                    'class_name': config["class_names"][class_idx]
                })
    
    print(f"Added {len(augmented_images)} augmented images")
    return (np.concatenate([images, np.array(augmented_images)]), 
            np.concatenate([np.array([a['class_idx'] for a in annotations]), np.array(augmented_labels)]),
            annotations + augmented_annotations)

def create_model():
    """
    Create a CNN model for prescription element classification
    """
    # Create directory for models
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    
    # Faster R-CNN style model
    input_img = Input(shape=(config["image_size"][0], config["image_size"][1], 3))
    
    # Feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Classification branch
    class_x = Flatten()(x)
    class_x = Dense(512, activation='relu')(class_x)
    class_x = Dropout(0.5)(class_x)
    class_output = Dense(len(config["class_names"]), activation='softmax', name='class_output')(class_x)
    
    # Bounding box regression branch
    bbox_x = Flatten()(x)
    bbox_x = Dense(512, activation='relu')(bbox_x)
    bbox_x = Dropout(0.5)(bbox_x)
    bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(bbox_x)  # [x, y, w, h] normalized
    
    # Create model
    model = Model(inputs=input_img, outputs=[class_output, bbox_output])
    
    # Compile model
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mse'
        },
        loss_weights={
            'class_output': 1.0,
            'bbox_output': 1.0
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mse'
        }
    )
    
    return model

def normalize_bbox(bbox, img_shape):
    """
    Normalize bounding box coordinates to [0, 1] range
    """
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    
    return [x/img_w, y/img_h, w/img_w, h/img_h]

def prepare_training_data(images, labels, annotations):
    """
    Prepare data for training
    """
    bbox_data = []
    
    for annot in annotations:
        bbox = annot['bbox']
        img_shape = images[0].shape  # Assuming all images have same shape after resize
        normalized_bbox = normalize_bbox(bbox, img_shape)
        bbox_data.append(normalized_bbox)
    
    bbox_data = np.array(bbox_data)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train_cls, y_temp_cls, y_train_bbox, y_temp_bbox = train_test_split(
        images, labels, bbox_data, test_size=config["validation_split"] + config["test_split"],
        random_state=42, stratify=labels)
    
    # Split temp into validation and test
    test_size_adjusted = config["test_split"] / (config["validation_split"] + config["test_split"])
    X_val, X_test, y_val_cls, y_test_cls, y_val_bbox, y_test_bbox = train_test_split(
        X_temp, y_temp_cls, y_temp_bbox, test_size=test_size_adjusted,
        random_state=42, stratify=y_temp_cls)
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return (X_train, y_train_cls, y_train_bbox,
            X_val, y_val_cls, y_val_bbox,
            X_test, y_test_cls, y_test_bbox)

def train_model(model, X_train, y_train_cls, y_train_bbox, X_val, y_val_cls, y_val_bbox):
    """
    Train the model
    """
    # Define callbacks
    checkpoint = ModelCheckpoint(
        config["model_save_path"],
        monitor='val_class_output_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_class_output_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_class_output_accuracy',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train,
        {'class_output': y_train_cls, 'bbox_output': y_train_bbox},
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_val, {'class_output': y_val_cls, 'bbox_output': y_val_bbox}),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test_cls, y_test_bbox):
    """
    Evaluate the model on test data
    """
    # Load best model
    try:
        model = load_model(config["model_save_path"])
    except:
        print("Could not load saved model. Using current model state.")
    
    # Evaluate model
    results = model.evaluate(
        X_test,
        {'class_output': y_test_cls, 'bbox_output': y_test_bbox},
        verbose=1
    )
    
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"{metric_name}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions[0], axis=1)
    pred_bboxes = predictions[1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_cls, pred_classes, 
                             target_names=config["class_names"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_cls, pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(config["class_names"]))
    plt.xticks(tick_marks, config["class_names"], rotation=45)
    plt.yticks(tick_marks, config["class_names"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    return predictions, pred_classes, pred_bboxes

def visualize_predictions(X_test, y_test_cls, pred_classes, pred_bboxes):
    """
    Visualize model predictions on test data
    """
    # Select random test samples
    indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(indices):
        # Original image
        img = X_test[idx] * 255.0
        img = img.astype(np.uint8)
        
        # True and predicted class
        true_class = config["class_names"][int(y_test_cls[idx])]
        pred_class = config["class_names"][int(pred_classes[idx])]
        
        # Predicted bounding box
        pred_bbox = pred_bboxes[idx]
        x, y, w, h = pred_bbox
        
        # De-normalize bounding box
        x = int(x * img.shape[1])
        y = int(y * img.shape[0])
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])
        
        # Draw predicted bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add class labels
        cv2.putText(img, f"True: {true_class}", (10, 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {pred_class}", (10, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show image
        plt.subplot(4, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample {idx}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def detect_elements_with_model(image_path, model=None):
    """
    Detect prescription elements in an image using the trained model
    """
    # Load the model if not provided
    if model is None:
        try:
            model = load_model(config["model_save_path"])
        except:
            print(f"Model not found at {config['model_save_path']}. Please train the model first.")
            return None, {}
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image at {image_path}")
        return None, {}
    
    original_height, original_width = img.shape[:2]
    processed_img = cv2.resize(img, config["image_size"])
    normalized_img = processed_img.astype('float32') / 255.0
    
    # Make prediction
    predictions = model.predict(np.expand_dims(normalized_img, axis=0))
    pred_class_probs = predictions[0][0]
    pred_bbox = predictions[1][0]
    
    # Get the class with highest probability
    pred_class_idx = np.argmax(pred_class_probs)
    pred_class = config["class_names"][pred_class_idx]
    confidence = pred_class_probs[pred_class_idx]
    
    # De-normalize bounding box to original image coordinates
    x_norm, y_norm, w_norm, h_norm = pred_bbox
    x = int(x_norm * original_width)
    y = int(y_norm * original_height)
    w = int(w_norm * original_width)
    h = int(h_norm * original_height)
    
    # Create copy of image for drawing
    img_with_boxes = img.copy()
    
    # Draw bounding box
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add class label
    cv2.putText(img_with_boxes, f"{pred_class} ({confidence:.2f})", 
              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Extract text from the region using OCR
    roi = img[y:y+h, x:x+w]
    if roi.size > 0:
        ocr_text = pytesseract.image_to_string(roi)
    else:
        ocr_text = ""
    
    detected_info = {
        'class': pred_class,
        'confidence': float(confidence),
        'bbox': [x, y, w, h],
        'text': ocr_text.strip()
    }
    
    return img_with_boxes, detected_info

def main():
    """
    Main function to run the prescription OCR model training pipeline
    """
    print("Prescription OCR Model Training Pipeline")
    print("=======================================")
    
    # Download and extract dataset
    print("\n1. Downloading dataset...")
    download_and_extract_dataset()
    
    # Load dataset
    print("\n2. Loading dataset...")
    images, labels, annotations = load_dataset()
    
    if images is None or labels is None or annotations is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Augment dataset
    print("\n3. Augmenting dataset...")
    images, labels, annotations = augment_dataset(images, labels, annotations)
    
    # Prepare training data
    print("\n4. Preparing training data...")
    (X_train, y_train_cls, y_train_bbox,
     X_val, y_val_cls, y_val_bbox,
     X_test, y_test_cls, y_test_bbox) = prepare_training_data(images, labels, annotations)
    
    # Create model
    print("\n5. Creating model...")
    model = create_model()
    model.summary()
    
    # Train model
    print("\n6. Training model...")
    history = train_model(model, X_train, y_train_cls, y_train_bbox, X_val, y_val_cls, y_val_bbox)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    predictions, pred_classes, pred_bboxes = evaluate_model(model, X_test, y_test_cls, y_test_bbox)
    
    # Visualize predictions
    print("\n8. Visualizing predictions...")
    visualize_predictions(X_test, y_test_cls, pred_classes, pred_bboxes)
    
    print("\nTraining pipeline completed!")
    
    # Demo on a sample image
    print("\n9. Running inference demo...")
    sample_files = list(Path(config["local_dataset_path"]).glob("**/*.jpg")) + \
                  list(Path(config["local_dataset_path"]).glob("**/*.png"))
    
    if sample_files:
        sample_image = str(random.choice(sample_files))
        print(f"Running inference on sample image: {sample_image}")
        img_with_boxes, detected_info = detect_elements_with_model(sample_image, model)
        
        if img_with_boxes is not None:
            cv2.imwrite("sample_detection.png", img_with_boxes)
            print(f"Detection results saved to sample_detection.png")
            print(f"Detected {detected_info['class']} with confidence {detected_info['confidence']:.2f}")
            print(f"Extracted text: {detected_info['text']}")
    else:
        print("No sample files found for demo.")
    
    print("\nModel saved to:", config["model_save_path"])
    print("Done!")

def integrated_prescription_detector():
    """
    Integrate the trained model with OpenCV and Tesseract for complete prescription processing
    """
    # Ensure model is trained and loaded
    if not os.path.exists(config["model_save_path"]):
        print(f"Model not found at {config['model_save_path']}. Please train the model first.")
        return
    
    try:
        model = load_model(config["model_save_path"])
        print("Model loaded successfully.")
    except:
        print("Error loading model. Please train the model first.")
        return
    
    class PrescriptionProcessor:
        def __init__(self, model):
            self.model = model
            self.class_names = config["class_names"]
            
            # Define colors for each element type
            self.colors = {
                'hospital': (255, 0, 0),    # Blue
                'patient': (0, 255, 0),     # Green
                'date': (0, 0, 255),        # Red
                'medicine': (255, 255, 0),  # Cyan
                'dosage': (255, 0, 255),    # Magenta
                'usage': (0, 255, 255),     # Yellow
                'doctor': (128, 128, 0)     # Olive
            }
        
        def preprocess_image(self, image):
            """Preprocess image for better OCR results"""
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return opening
        
        def sliding_window_detect(self, image, window_sizes=None):
            """Detect elements using sliding window approach"""
            if window_sizes is None:
                # Default window sizes as percentages of image dimensions
                window_sizes = [(0.5, 0.2), (0.3, 0.3), (0.7, 0.15)]
            
            h, w = image.shape[:2]
            
            # Resize image for model input
            resized_img = cv2.resize(image, config["image_size"])
            normalized_img = resized_img.astype('float32') / 255.0
            
            # Store all detections
            all_detections = []
            
            # Process each window size
            for w_perc, h_perc in window_sizes:
                window_w = int(w * w_perc)
                window_h = int(h * h_perc)
                
                # Define step size (overlap between windows)
                step_x = int(window_w * 0.5)
                step_y = int(window_h * 0.5)
                
                # Slide window over the image
                for y in range(0, h - window_h + 1, step_y):
                    for x in range(0, w - window_w + 1, step_x):
                        # Extract window
                        window = image[y:y+window_h, x:x+window_w]
                        
                        # Resize window to model input size
                        window_resized = cv2.resize(window, config["image_size"])
                        window_normalized = window_resized.astype('float32') / 255.0
                        
                        # Make prediction
                        predictions = self.model.predict(np.expand_dims(window_normalized, axis=0), verbose=0)
                        pred_class_probs = predictions[0][0]
                        pred_class_idx = np.argmax(pred_class_probs)
                        confidence = pred_class_probs[pred_class_idx]
                        
                        # Skip background class and low confidence detections
                        if pred_class_idx == 0 or confidence < 0.5:
                            continue
                            
                        pred_class = self.class_names[pred_class_idx]
                        
                        # Extract text using OCR
                        text = pytesseract.image_to_string(window)
                        
                        # Store detection
                        all_detections.append({
                            'class': pred_class,
                            'confidence': float(confidence),
                            'bbox': [x, y, window_w, window_h],
                            'text': text.strip()
                        })
            
            return all_detections
        
        def apply_non_max_suppression(self, detections, iou_threshold=0.5):
            """Apply non-max suppression to remove overlapping detections"""
            if not detections:
                return []
                
            # Sort by confidence
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize list of picked indices
            picked_detections = []
            
            while detections:
                # Pick the detection with highest confidence
                current = detections.pop(0)
                picked_detections.append(current)
                
                # Calculate IoU with remaining detections
                remaining_detections = []
                
                for det in detections:
                    # Only apply NMS to detections of the same class
                    if det['class'] != current['class']:
                        remaining_detections.append(det)
                        continue
                        
                    # Calculate IoU
                    x1, y1, w1, h1 = current['bbox']
                    x2, y2, w2, h2 = det['bbox']
                    
                    # Convert to top-left, bottom-right coordinates
                    current_rect = [x1, y1, x1 + w1, y1 + h1]
                    det_rect = [x2, y2, x2 + w2, y2 + h2]
                    
                    # Calculate intersection
                    x_left = max(current_rect[0], det_rect[0])
                    y_top = max(current_rect[1], det_rect[1])
                    x_right = min(current_rect[2], det_rect[2])
                    y_bottom = min(current_rect[3], det_rect[3])
                    
                    # No intersection
                    if x_right < x_left or y_bottom < y_top:
                        remaining_detections.append(det)
                        continue
                        
                    # Calculate areas
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    current_area = w1 * h1
                    det_area = w2 * h2
                    union_area = current_area + det_area - intersection_area
                    
                    # Calculate IoU
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou <= iou_threshold:
                        remaining_detections.append(det)
                
                detections = remaining_detections
            
            return picked_detections
        
        def process_image(self, image_path):
            """Process prescription image and detect all elements"""
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None, []
                
            # Make a copy for drawing
            img_with_boxes = image.copy()
            
            # Preprocess image for OCR
            preprocessed = self.preprocess_image(image)
            
            # Detect elements using sliding window
            detections = self.sliding_window_detect(image)
            
            # Apply non-max suppression
            filtered_detections = self.apply_non_max_suppression(detections)
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(filtered_detections):
                class_name = detection['class']
                confidence = detection['confidence']
                x, y, w, h = detection['bbox']
                text = detection['text']
                
                # Get color for this class
                color = self.colors.get(class_name, (0, 0, 0))
                
                # Draw bounding box
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(img_with_boxes, label, 
                          (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Group detections by class
            grouped_detections = {}
            for detection in filtered_detections:
                class_name = detection['class']
                if class_name not in grouped_detections:
                    grouped_detections[class_name] = []
                grouped_detections[class_name].append(detection)
            
            return img_with_boxes, grouped_detections
        
        def extract_structured_prescription(self, grouped_detections):
            """Extract structured prescription information from detections"""
            prescription = {
                'hospital': '',
                'patient': '',
                'date': '',
                'medicine': [],
                'dosage': [],
                'usage': [],
                'doctor': ''
            }
            
            # Process each detection group
            for class_name, detections in grouped_detections.items():
                if class_name in ['hospital', 'patient', 'date', 'doctor']:
                    # For single-value fields, use the highest confidence detection
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        prescription[class_name] = best_detection['text']
                elif class_name in ['medicine', 'dosage', 'usage']:
                    # For multi-value fields, collect all detections
                    for detection in detections:
                        prescription[class_name].append(detection['text'])
            
            # Normalize and clean up text
            for key in ['hospital', 'patient', 'date', 'doctor']:
                prescription[key] = self._clean_text(prescription[key])
                
            for key in ['medicine', 'dosage', 'usage']:
                prescription[key] = [self._clean_text(item) for item in prescription[key]]
            
            return prescription
        
        def _clean_text(self, text):
            """Clean and normalize text"""
            if not text:
                return ""
                
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove non-alphanumeric characters (except common punctuation)
            text = re.sub(r'[^\w\s.,;:()\-\']', '', text)
            
            return text
    
    # Initialize processor
    processor = PrescriptionProcessor(model)
    
    def process_prescription(image_path):
        """Process a prescription image and return results"""
        # Process image
        img_with_boxes, grouped_detections = processor.process_image(image_path)
        
        if img_with_boxes is None:
            return None, None
            
        # Extract structured prescription
        prescription_data = processor.extract_structured_prescription(grouped_detections)
        
        return img_with_boxes, prescription_data
    
    return process_prescription

# Integration with the Flask application
def integrate_with_flask_app():
    """
    Code to integrate the trained model with the Flask application
    """
    prescription_processor = integrated_prescription_detector()
    
    def detect_prescription_elements_with_model(image_path):
        """
        Process the image and detect prescription elements using the trained model
        Returns the image with bounding boxes and detected information
        """
        # Process the prescription image
        img_with_boxes, prescription_data = prescription_processor(image_path)
        
        if img_with_boxes is None:
            return None, {}
        
        # Format detected elements for compatibility with the original function
        detected_elements = {}
        for category, items in prescription_data.items():
            detected_elements[category] = []
            
            # Handle single string items
            if isinstance(items, str) and items:
                detected_elements[category].append({
                    'text': items,
                    'box': (0, 0, 0, 0)  # Placeholder, not used for display
                })
            # Handle list items
            elif isinstance(items, list):
                for item in items:
                    if item:
                        detected_elements[category].append({
                            'text': item,
                            'box': (0, 0, 0, 0)  # Placeholder, not used for display
                        })
        
        return img_with_boxes, detected_elements, prescription_data

    return detect_prescription_elements_with_model

# Flask app integration code
"""
To integrate this model with the Flask app, replace the detect_prescription_elements function 
in app.py with the following:

from prescription_model import integrate_with_flask_app

# Get the model-based detector
detect_prescription_elements_with_model = integrate_with_flask_app()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename to prevent overwriting
        unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the image
        try:
            img_with_boxes, detected_elements, prescription_data = detect_prescription_elements_with_model(filepath)
            
            # Save the processed image
            processed_filename = 'processed_' + unique_filename
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, img_with_boxes)
            
            # Save prescription data to session or database
            # session['prescription_data'] = prescription_data  # If using Flask session
            
            # Redirect to results page
            return redirect(url_for('results', 
                                    original=unique_filename, 
                                    processed=processed_filename))
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    else:
        flash('File type not allowed. Please upload a JPG, JPEG or PNG file.')
        return redirect(request.url)
"""

if __name__ == "__main__":
    main()