# app.py
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import re
import uuid
import json
from werkzeug.utils import secure_filename

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'prescription_ocr_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to remove noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Apply slight Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    # Save the preprocessed image for OCR
    preprocessed_path = image_path.replace('.', '_preprocessed.')
    cv2.imwrite(preprocessed_path, blurred)
    
    return preprocessed_path

def detect_prescription_elements(image_path):
    """
    Process the image and detect prescription elements using OCR
    Returns the image with bounding boxes and detected information
    """
    # Preprocess the image to improve OCR accuracy
    preprocessed_path = preprocess_image(image_path)
    
    # Read the original image for display purposes
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    
    # Make a copy for drawing bounding boxes
    img_with_boxes = img.copy()
    
    # Get OCR data with both the original and preprocessed images
    ocr_data_original = pytesseract.image_to_data(Image.open(image_path), 
                                                output_type=pytesseract.Output.DICT,
                                                config='--psm 11 --oem 3')
    
    ocr_data_preprocessed = pytesseract.image_to_data(Image.open(preprocessed_path), 
                                                   output_type=pytesseract.Output.DICT,
                                                   config='--psm 6 --oem 3')
    
    # Combine OCR results (remove duplicates later)
    combined_text = []
    for i in range(len(ocr_data_original['text'])):
        text = ocr_data_original['text'][i].strip()
        if text:
            combined_text.append({
                'text': text,
                'left': ocr_data_original['left'][i],
                'top': ocr_data_original['top'][i],
                'width': ocr_data_original['width'][i],
                'height': ocr_data_original['height'][i],
                'conf': ocr_data_original['conf'][i]
            })
    
    for i in range(len(ocr_data_preprocessed['text'])):
        text = ocr_data_preprocessed['text'][i].strip()
        if text:
            # Check if this is a duplicate (based on position and similar text)
            is_duplicate = False
            for existing in combined_text:
                if (abs(existing['left'] - ocr_data_preprocessed['left'][i]) < 20 and
                    abs(existing['top'] - ocr_data_preprocessed['top'][i]) < 20 and
                    (existing['text'].lower() in ocr_data_preprocessed['text'][i].lower() or
                     ocr_data_preprocessed['text'][i].lower() in existing['text'].lower())):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined_text.append({
                    'text': text,
                    'left': ocr_data_preprocessed['left'][i],
                    'top': ocr_data_preprocessed['top'][i],
                    'width': ocr_data_preprocessed['width'][i],
                    'height': ocr_data_preprocessed['height'][i],
                    'conf': ocr_data_preprocessed['conf'][i]
                })
    
    # Cleanup temporary preprocessed file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
    
    # Define more comprehensive patterns for better detection
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
    
    # Function to merge nearby boxes of the same category
    def merge_boxes(boxes, threshold=50):
        if not boxes:
            return []
        
        merged_boxes = []
        current_box = boxes[0]
        
        for box in boxes[1:]:
            x1, y1, w1, h1 = current_box['box']
            x2, y2, w2, h2 = box['box']
            
            # Check if boxes are close to each other
            if (abs(x1 - x2) < threshold and abs(y1 - y2) < threshold) or \
               (x1 <= x2 + w2 and x2 <= x1 + w1 and y1 <= y2 + h2 and y2 <= y1 + h1):
                # Merge boxes
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                
                current_box = {
                    'text': current_box['text'] + " " + box['text'],
                    'box': (x, y, w, h)
                }
            else:
                merged_boxes.append(current_box)
                current_box = box
        
        merged_boxes.append(current_box)
        return merged_boxes
    
    # Check text blocks for matching patterns
    for item in combined_text:
        text = item['text'].lower()
        x, y, w, h = item['left'], item['top'], item['width'], item['height']
        
        # Extended bounding box for context
        extended_x = max(0, x - 5)
        extended_y = max(0, y - 5)
        extended_w = min(width - extended_x, w + 10)
        extended_h = min(height - extended_y, h + 10)
        
        # Check against all patterns
        for element_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text):
                    detected_elements[element_type].append({
                        'text': text,
                        'box': (extended_x, extended_y, extended_w, extended_h)
                    })
                    break
    
    # Merge nearby boxes of the same category
    for element_type in detected_elements:
        detected_elements[element_type] = merge_boxes(detected_elements[element_type])
    
    # Define colors for each element type
    colors = {
        'hospital': (255, 0, 0),    # Blue
        'patient': (0, 255, 0),     # Green
        'date': (0, 0, 255),        # Red
        'medicine': (255, 255, 0),  # Cyan
        'dosage': (255, 0, 255),    # Magenta
        'usage': (0, 255, 255),     # Yellow
        'doctor': (128, 128, 0)     # Olive
    }
    
    # Create legend items
    legend_items = []
    
    # Draw the detected elements and collect legend information
    for element_type, elements in detected_elements.items():
        if elements:
            legend_items.append({
                'type': element_type.capitalize(),
                'count': len(elements),
                'color': colors[element_type]
            })
            
        for i, element in enumerate(elements):
            x, y, w, h = element['box']
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), colors[element_type], 2)
            
            # Add label above the box
            label = f"{element_type.capitalize()} {i+1}"
            cv2.putText(img_with_boxes, label, 
                      (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[element_type], 2)
    
    # Add legend to the image
    legend_y_start = 30
    for i, item in enumerate(legend_items):
        legend_text = f"{item['type']}: {item['count']} detected"
        cv2.putText(img_with_boxes, legend_text, 
                  (10, legend_y_start + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, item['color'], 2)
    
    # Extract the text for the detected elements for frontend display
    extracted_info = {}
    for element_type, elements in detected_elements.items():
        extracted_info[element_type] = [element['text'] for element in elements]
    
    return img_with_boxes, detected_elements, extracted_info

@app.route('/')
def index():
    return render_template('index.html')

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
            img_with_boxes, detected_elements, extracted_info = detect_prescription_elements(filepath)
            
            # Save the processed image
            processed_filename = 'processed_' + unique_filename
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, img_with_boxes)
            
            # Save extracted information in the session or pass it to template
            return redirect(url_for('results', 
                                    original=unique_filename, 
                                    processed=processed_filename,
                                    info=json.dumps(extracted_info)))
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    else:
        flash('File type not allowed. Please upload a JPG, JPEG or PNG file.')
        return redirect(request.url)

@app.route('/results')
def results():
    original = request.args.get('original')
    processed = request.args.get('processed')
    info = request.args.get('info')
    
    if not original or not processed:
        flash('Missing image information')
        return redirect(url_for('index'))
        
    original_path = url_for('static', filename=f'uploads/{original}')
    processed_path = url_for('static', filename=f'processed/{processed}')
    
    extracted_info = json.loads(info) if info else {}
    
    return render_template('results.html', 
                           original_image=original_path,
                           processed_image=processed_path,
                           extracted_info=extracted_info)

if __name__ == '__main__':
    app.run(debug=True)