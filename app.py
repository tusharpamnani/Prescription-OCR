# app.py
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import re
import uuid
import json
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'prescription_ocr_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'models/prescription_detector.h5'
app.config['TRAINING_MODE'] = False  # Set to True to enable model training

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Define class names
CLASS_NAMES = ["background", "hospital", "patient", "date", "medicine", "dosage", "usage", "doctor"]

# Colors for visualization
COLORS = {
    'hospital': 'rgb(255, 0, 0)',    # Blue
    'patient': 'rgb(0, 255, 0)',     # Green
    'date': 'rgb(0, 0, 255)',        # Red
    'medicine': 'rgb(255, 255, 0)',  # Cyan
    'dosage': 'rgb(255, 0, 255)',    # Magenta
    'usage': 'rgb(0, 255, 255)',     # Yellow
    'doctor': 'rgb(128, 128, 0)'     # Olive
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class PrescriptionDetector:
    def __init__(self):
        self.model = None
        self.image_size = (512, 512)
        try:
            self.load_model()
        except:
            print("Model not found or could not be loaded. Using fallback OCR method.")
    
    def load_model(self):
        """Load the trained model if available"""
        if os.path.exists(app.config['MODEL_PATH']):
            self.model = load_model(app.config['MODEL_PATH'])
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {app.config['MODEL_PATH']}")
            self.model = None
    
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
    
    def detect_elements_with_model(self, image_path):
        """Detect prescription elements using the trained model"""
        if self.model is None:
            print("Model not available, falling back to pattern-based detection")
            return self.detect_elements_with_patterns(image_path)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, {}, {}
        
        original_h, original_w = image.shape[:2]
        
        # Make a copy for drawing
        img_with_boxes = image.copy()
        
        # Store detections for each class
        detected_elements = {cls: [] for cls in CLASS_NAMES if cls != "background"}
        prescription_data = {cls: [] for cls in CLASS_NAMES if cls != "background"}
        
        # Define sliding window sizes (as percentages of image dimensions)
        window_sizes = [(0.5, 0.2), (0.3, 0.3), (0.7, 0.15)]
        
        # Process multiple window sizes
        for w_perc, h_perc in window_sizes:
            window_w = int(original_w * w_perc)
            window_h = int(original_h * h_perc)
            
            # Define step size (overlap between windows)
            step_x = int(window_w * 0.5)
            step_y = int(window_h * 0.5)
            
            # Slide window over the image
            for y in range(0, original_h - window_h + 1, step_y):
                for x in range(0, original_w - window_w + 1, step_x):
                    # Extract window
                    window = image[y:y+window_h, x:x+window_w]
                    
                    # Resize window to model input size
                    window_resized = cv2.resize(window, self.image_size)
                    window_normalized = window_resized.astype('float32') / 255.0
                    
                    # Make prediction
                    predictions = self.model.predict(np.expand_dims(window_normalized, axis=0), verbose=0)
                    pred_class_probs = predictions[0][0]
                    pred_bbox = predictions[1][0]  # Normalized bbox within the window
                    
                    pred_class_idx = np.argmax(pred_class_probs)
                    confidence = pred_class_probs[pred_class_idx]
                    
                    # Skip background class and low confidence detections
                    if pred_class_idx == 0 or confidence < 0.5:
                        continue
                        
                    # Get class name
                    pred_class = CLASS_NAMES[pred_class_idx]
                    
                    # Calculate bounding box within the window
                    rel_x, rel_y, rel_w, rel_h = pred_bbox
                    bbox_x = x + int(rel_x * window_w)
                    bbox_y = y + int(rel_y * window_h)
                    bbox_w = int(rel_w * window_w)
                    bbox_h = int(rel_h * window_h)
                    
                    # Ensure the bounding box doesn't exceed image dimensions
                    bbox_x = max(0, bbox_x)
                    bbox_y = max(0, bbox_y)
                    bbox_w = min(original_w - bbox_x, bbox_w)
                    bbox_h = min(original_h - bbox_y, bbox_h)
                    
                    # Extract text using OCR
                    roi = image[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
                    if roi.size > 0:
                        # Preprocess the ROI for better OCR
                        processed_roi = self.preprocess_image(roi)
                        text = pytesseract.image_to_string(processed_roi)
                        text = text.strip()
                        
                        if text:
                            # Add to detected elements
                            detected_elements[pred_class].append({
                                'text': text,
                                'box': (bbox_x, bbox_y, bbox_w, bbox_h),
                                'confidence': float(confidence)
                            })
                            
                            # Add to prescription data
                            prescription_data[pred_class].append(text)
        
        # Apply non-max suppression to remove overlapping detections of the same class
        for class_name in detected_elements:
            # Sort by confidence
            detected_elements[class_name] = sorted(
                detected_elements[class_name], 
                key=lambda x: x['confidence'], 
                reverse=True
            )
            
            # Apply non-max suppression
            kept_detections = []
            for detection in detected_elements[class_name]:
                # Check if this detection overlaps significantly with any kept detection
                should_keep = True
                for kept in kept_detections:
                    iou = self._calculate_iou(detection['box'], kept['box'])
                    if iou > 0.5:  # Threshold for considering as overlap
                        should_keep = False
                        break
                
                if should_keep:
                    kept_detections.append(detection)
            
            detected_elements[class_name] = kept_detections
        
        # Draw bounding boxes on the image
        for class_name, elements in detected_elements.items():
            color = COLORS.get(class_name, (0, 0, 0))
            
            for i, element in enumerate(elements):
                x, y, w, h = element['box']
                confidence = element.get('confidence', 0)
                
                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(img_with_boxes, label, 
                          (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Clean up prescription data
        for class_name in prescription_data:
            # For single value fields, keep only the highest confidence detection
            if class_name in ['hospital', 'patient', 'date', 'doctor']:
                if prescription_data[class_name] and detected_elements[class_name]:
                    # Get text from highest confidence detection
                    best_detection = detected_elements[class_name][0]
                    prescription_data[class_name] = best_detection['text']
                else:
                    prescription_data[class_name] = ""
        
        # Add legend
        y_offset = 30
        for class_name, color in COLORS.items():
            if class_name in detected_elements and detected_elements[class_name]:
                count = len(detected_elements[class_name])
                cv2.putText(img_with_boxes, f"{class_name}: {count} detected", 
                          (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
        
        return img_with_boxes, detected_elements, prescription_data
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates of intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check if there is intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def detect_elements_with_patterns(self, image_path):
        """
        Fallback method: Process the image and detect prescription elements using pattern matching
        Returns the image with bounding boxes and detected information
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None, {}, {}
            
        height, width, _ = img.shape
        
        # Make a copy for drawing bounding boxes
        img_with_boxes = img.copy()
        
        # Convert to PIL Image for Tesseract
        pil_img = Image.open(image_path)
        
        # Get OCR data with bounding box information
        ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        
        # Define patterns to match
        patterns = {
            'hospital': [
                r'hospital', r'medical center', r'clinic', r'healthcare', r'health center',
                r'medical', r'health', r'care', r'center', r'institute', r'department',
                r'memorial', r'university', r'community', r'regional'
            ],
            'patient': [
                r'patient name', r'name of patient', r'patient', r'name', r'pt\.?', r'pt name',
                r'patient information', r'client', r'patient id', r'mrn', r'chart'
            ],
            'date': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'date', r'issued on', r'prescription date',
                r'dob', r'birth date', r'date of birth', r'date issued', r'prescribed on',
                r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}'
            ],
            'medicine': [
                r'rx', r'medicine', r'drug', r'tablet', r'capsule', r'mg', r'ml', r'mcg',
                r'prescription', r'prescribed', r'medication', r'tabs', r'cap', r'solution',
                r'susp', r'injection', r'cream', r'ointment', r'lotion', r'drops', r'spray',
                r'inhaler', r'patch', r'syrup', r'elixir', r'suppository'
            ],
            'dosage': [
                r'\d+\s*mg', r'\d+\s*ml', r'\d+\s*mcg', r'\d+\s*tab', r'\d+\s*pill',
                r'\d+\s*cap', r'\d+\s*dose', r'\d+\s*puff', r'\d+\s*unit', r'every\s+\d+\s*hours?',
                r'\d+\s*times?\s+daily', r'\d+\s*times?\s+a\s+day'
            ],
            'usage': [
                r'take', r'use', r'apply', r'administer', r'inject', r'inhale', r'chew', r'dissolve',
                r'swallow', r'once daily', r'twice daily', r'three times daily', r'four times daily',
                r'every morning', r'every night', r'before meals?', r'after meals?', r'with food',
                r'without food', r'as needed', r'when required', r'p\.r\.n\.', r'prn',
                r'with water', r'orally', r'topically', r'externally', r'intramuscular',
                r'intravenous', r'subcutaneous', r'directions', r'instructions'
            ],
            'doctor': [
                r'dr\.', r'doctor', r'physician', r'prescribed by', r'signature', r'md',
                r'prescriber', r'practitioner', r'provider', r'do', r'pa', r'np',
                r'license', r'registration', r'reg\.', r'no\.', r'number'
            ]
        }
        
        # Store detected elements
        detected_elements = {k: [] for k in patterns.keys()}
        prescription_data = {k: [] for k in patterns.keys()}
        
        # Check each OCR detected text
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].lower().strip()
            if not text:
                continue
                
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            # Check if text matches any pattern
# Check if text matches any pattern
            for element_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, text, re.IGNORECASE):
                        detected_elements[element_type].append({
                            'text': text,
                            'box': (x, y, w, h)
                        })
                        prescription_data[element_type].append(text)
                        
                        # Draw rectangle
                        color = COLORS.get(element_type, (0, 0, 0))
                        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw label
                        cv2.putText(img_with_boxes, element_type, 
                                   (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        break  # Stop checking patterns for this element type
        
        # Clean up prescription data
        for element_type in prescription_data:
            # Join all detected text for each element type
            prescription_data[element_type] = ' '.join(prescription_data[element_type])
        
        # Add legend
        y_offset = 30
        for element_type, color in COLORS.items():
            if element_type in detected_elements and detected_elements[element_type]:
                count = len(detected_elements[element_type])
                cv2.putText(img_with_boxes, f"{element_type}: {count} detected", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
        
        return img_with_boxes, detected_elements, prescription_data
    
    def store_training_data(self, image_path, annotations):
        """Store annotated data for model training"""
        # Generate a unique filename for the annotations
        image_name = os.path.basename(image_path)
        annotation_file = os.path.join('models', f"{os.path.splitext(image_name)[0]}_annotations.json")
        
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        print(f"Training data stored in {annotation_file}")
        return annotation_file


# Initialize the detector
detector = PrescriptionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # Save the original file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the image
        try:
            if detector.model is not None:
                processed_img, detected_elements, prescription_data = detector.detect_elements_with_model(filepath)
            else:
                processed_img, detected_elements, prescription_data = detector.detect_elements_with_patterns(filepath)
            
            if processed_img is None:
                flash('Failed to process image')
                return redirect(url_for('index'))
            
            # Save the processed image
            processed_filename = f"processed_{unique_filename}"
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_img)
            
            # Store data in session
            session['uploaded_img'] = unique_filename
            session['processed_img'] = processed_filename
            session['detected_elements'] = detected_elements
            session['prescription_data'] = prescription_data
            
            return redirect(url_for('results'))
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Please upload an image file (png, jpg, jpeg)')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'uploaded_img' not in session or 'processed_img' not in session:
        flash('Please upload an image first')
        return redirect(url_for('index'))
    
    uploaded_img = session['uploaded_img']
    processed_img = session['processed_img']
    detected_elements = session.get('detected_elements', {})
    prescription_data = session.get('prescription_data', {})
    
    return render_template('results.html', 
                           uploaded_img=uploaded_img,
                           processed_img=processed_img,
                           detected_elements=detected_elements,
                           prescription_data=prescription_data)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # Save the original file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the image
        try:
            if detector.model is not None:
                _, detected_elements, prescription_data = detector.detect_elements_with_model(filepath)
            else:
                _, detected_elements, prescription_data = detector.detect_elements_with_patterns(filepath)
            
            # Format the response
            response = {
                'success': True,
                'detected_elements': detected_elements,
                'prescription_data': prescription_data
            }
            
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload an image file (png, jpg, jpeg)'}), 400

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if not app.config['TRAINING_MODE']:
        flash('Training mode is disabled')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        annotations = request.json
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_img'])
        
        detector.store_training_data(image_path, annotations)
        return jsonify({'success': True})
    
    if 'uploaded_img' not in session:
        flash('Please upload an image first')
        return redirect(url_for('index'))
    
    return render_template('annotate.html', uploaded_img=session['uploaded_img'])

@app.route('/train', methods=['POST'])
def train_model():
    if not app.config['TRAINING_MODE']:
        return jsonify({'error': 'Training mode is disabled'}), 403
    
    # In a real implementation, we would run model training here
    return jsonify({'message': 'Model training initiated. This would take some time in a real implementation.'})

@app.route('/download_data', methods=['GET'])
def download_data():
    if 'prescription_data' not in session:
        flash('No prescription data available')
        return redirect(url_for('index'))
    
    # Format the data as needed for the client
    data = session.get('prescription_data', {})
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)