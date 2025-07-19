from flask import Flask, render_template, request, redirect, url_for
import pytesseract
from PIL import Image
import os
import spacy
import pandas as pd
import cv2
import numpy as np
import re
import requests
from urllib.parse import quote_plus

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Specify Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"D:\6th sem pbl\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# Create upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create user uploads folder for store medicines
STORE_UPLOAD_FOLDER = 'static/store_meds'
os.makedirs(STORE_UPLOAD_FOLDER, exist_ok=True)

# Load medicine dataset and create lookup dictionary
MEDICINE_DATA_PATH = "static/A_Z_medicines_dataset_of_India.csv"
medicine_df = pd.read_csv(MEDICINE_DATA_PATH)

medicine_lookup = {}
for _, row in medicine_df.iterrows():
    med_name = row['name'].lower()
    composition = f"{str(row['short_composition1'])}, {str(row['short_composition2'])}" if pd.notna(row['short_composition2']) else str(row['short_composition1'])
    medicine_lookup[med_name] = {
        "name": row['name'],
        "price": row['price(₹)'],
        "manufacturer": row['manufacturer_name'],
        "pack": row['pack_size_label'],
        "composition": composition
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reports')
def reports_page():
    return render_template('Report.html')


@app.route('/scan', methods=['POST'])
def scan():
    uploaded_file = request.files['prescription']
    if uploaded_file and uploaded_file.filename != '':
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(filepath)

        # Enhance image for better OCR
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        blur = cv2.medianBlur(thresh, 3)
        processed_path = os.path.join(UPLOAD_FOLDER, "processed.png")
        cv2.imwrite(processed_path, blur)

        # OCR
        text = pytesseract.image_to_string(Image.open(processed_path))

        # Extract medicine names and dosage
        medicines = extract_medicines(text)
        dosages = extract_dosage_info(text)

        # Match and enrich medicine details
        medicine_details = []
        for med in medicines:
            med_lower = med.lower()
            found = None
            for key in medicine_lookup:
                if med_lower in key:
                    found = medicine_lookup[key]
                    break
            if found:
                medicine_details.append(found)
            else:
                medicine_details.append({
                    "name": med,
               #     "price": "Not found",
                #    "manufacturer": "Not found",
                 #   "pack": "Not found",
                  #  "composition": "Not found"
                })

        # Store prescription info in session
        session_data = {
            'text': text,
            'medicine_details': medicine_details,
            'dosages': dosages,
            'prescription_path': filepath
        }
        
        # Save session data to a file (simple approach for demo)
        import json
        with open('session_data.json', 'w') as f:
            json.dump(session_data, f, default=str)

        return render_template('index.html', 
                               text=text, 
                               medicine_details=medicine_details, 
                               dosages=dosages,
                               prescription_path=os.path.basename(filepath))

    return 'No file uploaded or invalid file format!'

@app.route('/verify', methods=['GET', 'POST'])
def verify_medicine():
    if request.method == 'POST':
        # Get uploaded store medicine image
        store_med_file = request.files.get('store_medicine')
        medicine_name = request.form.get('medicine_name')
        
        if store_med_file and store_med_file.filename != '':
            # Save the store medicine image
            store_med_path = os.path.join(STORE_UPLOAD_FOLDER, store_med_file.filename)
            store_med_file.save(store_med_path)
            
            # Perform OCR on store medicine
            store_text = pytesseract.image_to_string(Image.open(store_med_path))
            
            # Try to load session data
            import json
            try:
                with open('session_data.json', 'r') as f:
                    session_data = json.load(f)
            except:
                session_data = {}
            
            # Extract medicine details from prescription
            prescription_medicines = session_data.get('medicine_details', [])
            
            # Find the selected medicine in our records
            selected_medicine = None
            for med in prescription_medicines:
                if med['name'].lower() == medicine_name.lower():
                    selected_medicine = med
                    break
            
            # If we couldn't find it, try to look it up
            if not selected_medicine:
                for key, value in medicine_lookup.items():
                  if medicine_name.lower() in key:
                       selected_medicine = value
                       break
            
            # If still not found, create a placeholder
            if not selected_medicine:
                selected_medicine = {
                    "name": medicine_name,
                   "price": "Not found",
                #    "manufacturer": "Not found",
                 #   "pack": "Not found",
                  #  "composition": "Not found"
                }
            
            # Analyze if store medicine matches prescription
            # Simple check if medicine name is in the OCR text
            verification_result = {
                "name_match": medicine_name.lower() in store_text.lower(),
                "store_text": store_text,
                "store_image": os.path.basename(store_med_path),
                "selected_medicine": selected_medicine
            }
            
            # Get additional information from online resources
            search_query = f"{medicine_name} medicine information"
            search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"
            
            return render_template('verify.html', 
                                  verification=verification_result,
                                  search_url=search_url)
        
        return 'No medicine image uploaded!'
    
    # GET request - show the verification form
    # Try to load session data to show available medicines
    import json
    try:
        with open('session_data.json', 'r') as f:
            session_data = json.load(f)
            medicines = [med['name'] for med in session_data.get('medicine_details', [])]
    except:
        medicines = []
    
    return render_template('verify.html', medicines=medicines)

def extract_medicines(text):
    doc = nlp(text)
    medicines = []
    for ent in doc.ents:
        if ent.label_ == "PRODUCT":
            medicines.append(ent.text)
    
    # Fallback: look for capitalized words that might be medicine names
    if not medicines:
        lines = text.split('\n')
        for line in lines:
            words = line.strip().split()
            for word in words:
                if len(word) > 3 and word[0].isupper():
                    if word.lower() not in ['take', 'daily', 'once', 'twice', 'tablet', 'capsule']:
                        medicines.append(word)
    
    return medicines

def extract_dosage_info(text):
    dosage_patterns = [
        r'\b\d[- ]\d[- ]\d\b',                    # e.g., 1-0-1
        r'\btwice a day\b',
        r'\bonce daily\b',
        r'\bthree times a day\b',
        r'\b1\s?x\s?1\b',
        r'\b2\s?x\s?1\b',
        r'\bevery\s+\d+\s+(hours|hrs|hr)\b'
    ]
    matches = []
    for pattern in dosage_patterns:
        matches += re.findall(pattern, text, flags=re.IGNORECASE)
    return matches

if __name__ == '__main__':
    app.run(debug=True)