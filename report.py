from flask import Flask, render_template, request, redirect, url_for, session
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
import json
from collections import Counter
import nltk
# Add PDF processing libraries
import PyPDF2
import io

# Initialize SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if SpaCy model is not available
    import en_core_web_sm
    nlp = en_core_web_sm.load()

# Specify Tesseract path (adjust if needed)
# Comment out this line if Tesseract is in system PATH
pytesseract.pytesseract.tesseract_cmd = r"D:\6th sem pbl\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = "medical_report_analyzer_secret_key"  # Required for session

# Create upload folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create user uploads folder for store medicines
STORE_UPLOAD_FOLDER = 'static/store_meds'
os.makedirs(STORE_UPLOAD_FOLDER, exist_ok=True)

# Create folder for medical reports
REPORTS_FOLDER = 'static/reports'
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass  # Handle the case where NLTK data might already be downloaded

# Load medicine dataset and create lookup dictionary
MEDICINE_DATA_PATH = "static/A_Z_medicines_dataset_of_India.csv"
medicine_lookup = {}

# Try to load medicine dataset if available
try:
    medicine_df = pd.read_csv(MEDICINE_DATA_PATH)
    
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
except Exception as e:
    print(f"Warning: Could not load medicine dataset: {e}")

@app.route('/')
def index():
    return render_template('Report.html')

@app.route('/scan', methods=['POST'])
def scan():
    try:
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
                        "price": "Not found",
                        "manufacturer": "Not found",
                        "pack": "Not found",
                        "composition": "Not found"
                    })

            # Store prescription info in session
            session['prescription_data'] = {
                'text': text,
                'medicine_details': medicine_details,
                'dosages': dosages,
                'prescription_path': filepath
            }

            return render_template('Report.html', 
                                text=text, 
                                medicine_details=medicine_details, 
                                dosages=dosages,
                                prescription_path=os.path.basename(filepath))
                                
    except Exception as e:
        return render_template('Report.html', error=f"Error processing prescription: {str(e)}")

    return render_template('Report.html', error='No file uploaded or invalid file format!')

@app.route('/verify', methods=['GET', 'POST'])
def verify_medicine():
    if request.method == 'POST':
        try:
            # Get uploaded store medicine image
            store_med_file = request.files.get('store_medicine')
            medicine_name = request.form.get('medicine_name')
            
            if store_med_file and store_med_file.filename != '':
                # Save the store medicine image
                store_med_path = os.path.join(STORE_UPLOAD_FOLDER, store_med_file.filename)
                store_med_file.save(store_med_path)
                
                # Perform OCR on store medicine
                store_text = pytesseract.image_to_string(Image.open(store_med_path))
                
                # Try to get prescription data from session
                prescription_data = session.get('prescription_data', {})
                prescription_medicines = prescription_data.get('medicine_details', [])
                
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
                        "manufacturer": "Not found",
                        "pack": "Not found",
                        "composition": "Not found"
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
                
                return render_template('Report.html', 
                                    verification=verification_result,
                                    search_url=search_url)
        except Exception as e:
            return render_template('Report.html', error=f"Error verifying medicine: {str(e)}")
        
        return render_template('Report.html', error='No medicine image uploaded!')
    
    # GET request - show the verification form
    prescription_data = session.get('prescription_data', {})
    medicines = [med['name'] for med in prescription_data.get('medicine_details', [])]
    
    return render_template('Report.html', medicines=medicines)

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

@app.route('/reports')
def reports_page():
    return render_template('Report.html')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error extracting text from PDF: {str(e)}"

@app.route('/analyze_report', methods=['POST'])
def analyze_report():
    try:
        uploaded_file = request.files['medical_report']
        if uploaded_file and uploaded_file.filename != '':
            # Save the uploaded report
            filepath = os.path.join(REPORTS_FOLDER, uploaded_file.filename)
            uploaded_file.save(filepath)
            
            # Extract text from the report based on file type
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                # For image files, use OCR
                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
                blur = cv2.medianBlur(thresh, 3)
                processed_path = os.path.join(REPORTS_FOLDER, "processed_report.png")
                cv2.imwrite(processed_path, blur)
                
                report_text = pytesseract.image_to_string(Image.open(processed_path))
            elif filepath.lower().endswith('.pdf'):
                # For PDF files, use PyPDF2
                report_text = extract_text_from_pdf(filepath)
                
                # If the PDF might be image-based (scanned document), also try OCR as fallback
                if not report_text or len(report_text.strip()) < 100:
                    try:
                        # Convert the first page of the PDF to an image and use OCR
                        images = convert_pdf_to_images(filepath)
                        if images:
                            ocr_texts = []
                            for i, img in enumerate(images):
                                img_path = os.path.join(REPORTS_FOLDER, f"temp_pdf_page_{i}.png")
                                img.save(img_path)
                                
                                # Process the image for better OCR
                                cv_img = cv2.imread(img_path)
                                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, 11, 2)
                                blur = cv2.medianBlur(thresh, 3)
                                processed_path = os.path.join(REPORTS_FOLDER, f"processed_pdf_page_{i}.png")
                                cv2.imwrite(processed_path, blur)
                                
                                page_text = pytesseract.image_to_string(Image.open(processed_path))
                                ocr_texts.append(page_text)
                                
                                # Clean up temporary files
                                try:
                                    os.remove(img_path)
                                    os.remove(processed_path)
                                except:
                                    pass
                                
                            report_text = "\n\n".join(ocr_texts)
                    except Exception as e:
                        print(f"Error during PDF-to-image OCR: {e}")
                        # Keep whatever text we extracted with PyPDF2
            else:
                # For text files
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    report_text = f.read()
            
            # Analyze the report with enhanced functionality
            analysis_results = analyze_medical_report(report_text)
            summary = generate_report_summary(report_text)
            medical_terms = extract_medical_terms(report_text)

                        
            # Store the analysis results in session for reference
            session['report_analysis'] = {
                'text': report_text,
                'analysis': analysis_results,
                'report_path': filepath
            }

            # Return the results
            return render_template('Report.html', 
                                summary=summary,
                                abnormal_values=analysis_results['abnormal_values'],
                                health_issues=analysis_results['health_issues'],
                                recommendations=analysis_results['recommendations'],
                                risk_level=analysis_results['risk_level'],
                                abbreviations=medical_terms,
                                report_text=report_text,
                                report_path=os.path.basename(filepath))
    except Exception as e:
        return render_template('Report.html', error=f"Error analyzing report: {str(e)}")
    
    return render_template('Report.html', error='No file uploaded or invalid file format!')

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to images for OCR processing"""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path)
    except ImportError:
        print("pdf2image library not installed. Cannot convert PDF to images for OCR.")
        return []
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def analyze_medical_report(report_text):
    """
    Analyzes a medical report to identify abnormal values, health issues, and recommended actions
    
    Args:
        report_text (str): The text content of the medical report
        
    Returns:
        dict: Dictionary containing analysis results with the following keys:
            - abnormal_values: List of abnormal test values detected
            - health_issues: List of potential health issues identified
            - recommendations: List of suggested actions or follow-ups
            - risk_level: Overall risk assessment (Low, Moderate, High)
    """
    results = {
        'abnormal_values': [],
        'health_issues': [],
        'recommendations': [],
        'risk_level': 'Low'
    }
    
    # Extract abnormal values using existing and enhanced patterns
    results['abnormal_values'] = identify_abnormal_values(report_text)
    
    # Identify potential health issues based on abnormal values and keywords
    results['health_issues'] = identify_health_issues(report_text, results['abnormal_values'])
    
    # Generate recommendations based on identified issues
    results['recommendations'] = generate_recommendations(results['health_issues'])
    
    # Determine overall risk level
    results['risk_level'] = assess_risk_level(results['abnormal_values'], results['health_issues'])
    
    return results

def identify_abnormal_values(text):
    """Enhanced version of the existing function to detect abnormal values in medical reports"""
    import re
    
    # More comprehensive patterns for detecting abnormal values
    abnormal_patterns = [
        # Standard notation patterns
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)(H|HIGH|High|↑|h|abnormal|ABNORMAL)',
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)(L|LOW|Low|↓|l|abnormal|ABNORMAL)',
        
        # Patterns with parentheses
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)\(?(H|HIGH|High|↑|h|abnormal|ABNORMAL)',
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)\(?(L|LOW|Low|↓|l|abnormal|ABNORMAL)',
        
        # Patterns with reference ranges
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)(?:.*?ref\.?\s*range:?\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*))',
        
        # Patterns with flags
        r'(\w+(?:\s\w+)?)\s*[:]\s*(\d+\.?\d*)\s*[*]?(?:\s*)(?:.*?Flag:?\s*(H|HIGH|High|↑|h|L|LOW|Low|↓|l|abnormal|ABNORMAL))',
        
        # Common abnormal notations
        r'(\w+(?:\s\w+)?):?\s*(?:is\s*)?(\d+\.?\d*)\s*[*]?(?:\s*)(?:.*?(?:high|low|elevated|decreased|abnormal))',
    ]
    
    # Special test patterns with qualitative results
    qualitative_patterns = [
        r'(\w+(?:\s\w+)?)\s*[:]\s*(positive|negative|reactive|non-reactive|detected|not detected|present|absent)\s*[*]?(?:\s*)(?:.*?(?:abnormal|concerning|requires attention))',
    ]
    
    # Common lab tests with their normal ranges for numeric validation
    normal_ranges = {
        'hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 15.5), 'units': 'g/dL'},
        'wbc': {'range': (4.5, 11.0), 'units': '10^3/μL'},
        'rbc': {'range': (4.5, 5.9), 'units': '10^6/μL'},
        'platelets': {'range': (150, 450), 'units': '10^3/μL'},
        'hematocrit': {'male': (41, 50), 'female': (36, 44), 'units': '%'},
        'glucose': {'fasting': (70, 99), 'random': (70, 140), 'units': 'mg/dL'},
        'creatinine': {'male': (0.7, 1.3), 'female': (0.6, 1.1), 'units': 'mg/dL'},
        'bun': {'range': (7, 20), 'units': 'mg/dL'},
        'alt': {'range': (7, 56), 'units': 'U/L'},
        'ast': {'range': (10, 40), 'units': 'U/L'},
        'total cholesterol': {'range': (125, 200), 'units': 'mg/dL'},
        'hdl': {'range': (40, 60), 'units': 'mg/dL'},
        'ldl': {'range': (0, 100), 'units': 'mg/dL'},
        'triglycerides': {'range': (0, 150), 'units': 'mg/dL'},
        'tsh': {'range': (0.4, 4.0), 'units': 'mIU/L'},
        'sodium': {'range': (135, 145), 'units': 'mmol/L'},
        'potassium': {'range': (3.5, 5.0), 'units': 'mmol/L'},
        'calcium': {'range': (8.5, 10.5), 'units': 'mg/dL'},
        'vitamin d': {'range': (30, 100), 'units': 'ng/mL'},
        'vitamin b12': {'range': (200, 900), 'units': 'pg/mL'},
        'a1c': {'range': (4.0, 5.6), 'units': '%'},
    }
    
    # Results collection
    abnormal_values = []
    
    # Process numeric patterns
    for pattern in abnormal_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                test_name = match.group(1).strip().lower()
                value_str = match.group(2).strip()
                
                try:
                    value = float(value_str)
                    
                    # Check if we have normal ranges for this test
                    is_abnormal = False
                    direction = ""
                    
                    if len(match.groups()) >= 3 and match.group(3):
                        flag = match.group(3).upper()
                        if flag in ['H', 'HIGH', '↑', 'ABNORMAL']:
                            is_abnormal = True
                            direction = "high"
                        elif flag in ['L', 'LOW', '↓', 'ABNORMAL']:
                            is_abnormal = True
                            direction = "low"
                    
                    # Double check against known normal ranges if available
                    if test_name in normal_ranges:
                        range_info = normal_ranges[test_name]
                        units = range_info.get('units', '')
                        
                        if 'range' in range_info:
                            min_val, max_val = range_info['range']
                            if value < min_val:
                                is_abnormal = True
                                direction = "low"
                            elif value > max_val:
                                is_abnormal = True
                                direction = "high"
                        # Gender-specific ranges would need gender info from the report
                    
                    if is_abnormal:
                        test_display = test_name.upper() if test_name.isupper() else test_name.title()
                        abnormal_values.append({
                            'test': test_display,
                            'value': value,
                            'direction': direction,
                            'units': normal_ranges.get(test_name, {}).get('units', '')
                        })
                except ValueError:
                    # Not a numeric value, skip
                    pass
    
    # Process qualitative patterns
    for pattern in qualitative_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                test_name = match.group(1).strip().lower()
                result = match.group(2).strip().lower()
                
                # Determine if result is concerning based on common tests
                is_abnormal = False
                if result in ['positive', 'reactive', 'detected', 'present']:
                    # For tests where positive is usually abnormal
                    if any(term in test_name.lower() for term in ['covid', 'hiv', 'hepatitis', 'bacteria', 'infection']):
                        is_abnormal = True
                elif result in ['negative', 'non-reactive', 'not detected', 'absent']:
                    # For tests where negative could be abnormal
                    if any(term in test_name.lower() for term in ['antibody', 'vitamin', 'protein']):
                        is_abnormal = True
                
                if is_abnormal:
                    test_display = test_name.upper() if test_name.isupper() else test_name.title()
                    abnormal_values.append({
                        'test': test_display,
                        'value': result,
                        'direction': 'abnormal',
                        'units': ''
                    })
    
    return abnormal_values

def identify_health_issues(text, abnormal_values):
    """
    Identifies potential health issues based on abnormal values and medical keywords
    
    Args:
        text (str): The text content of the medical report
        abnormal_values (list): List of detected abnormal values
        
    Returns:
        list: Potential health issues identified from the report
    """
    import re
    
    # Dictionary mapping health conditions to related tests and keywords
    health_conditions = {
        'Anemia': {
            'tests': ['hemoglobin', 'rbc', 'hematocrit', 'mch', 'mchc', 'ferritin', 'iron'],
            'low_tests': ['hemoglobin', 'rbc', 'hematocrit', 'mch', 'mchc', 'ferritin', 'iron'],
            'high_tests': [],
            'keywords': ['anemia', 'iron deficiency', 'pale', 'fatigue', 'shortness of breath', 'weakness']
        },
        'Diabetes': {
            'tests': ['glucose', 'a1c', 'blood sugar', 'fasting glucose', 'random glucose'],
            'low_tests': [],
            'high_tests': ['glucose', 'a1c', 'blood sugar', 'fasting glucose', 'random glucose'],
            'keywords': ['diabetes', 'hyperglycemia', 'polyuria', 'polydipsia', 'polyphagia']
        },
        'Hyperlipidemia': {
            'tests': ['cholesterol', 'ldl', 'triglycerides', 'total cholesterol'],
            'low_tests': ['hdl'],
            'high_tests': ['cholesterol', 'ldl', 'triglycerides', 'total cholesterol'],
            'keywords': ['hyperlipidemia', 'hypercholesterolemia', 'high cholesterol']
        },
        'Hypertension': {
            'tests': ['blood pressure', 'bp', 'systolic', 'diastolic'],
            'low_tests': [],
            'high_tests': ['blood pressure', 'bp', 'systolic', 'diastolic'],
            'keywords': ['hypertension', 'high blood pressure', 'elevated blood pressure']
        },
        'Hypothyroidism': {
            'tests': ['tsh', 't4', 't3', 'thyroid'],
            'low_tests': ['t4', 't3'],
            'high_tests': ['tsh'],
            'keywords': ['hypothyroidism', 'thyroid', 'fatigue', 'cold intolerance', 'weight gain']
        },
        'Hyperthyroidism': {
            'tests': ['tsh', 't4', 't3', 'thyroid'],
            'low_tests': ['tsh'],
            'high_tests': ['t4', 't3'],
            'keywords': ['hyperthyroidism', 'thyroid', 'weight loss', 'heat intolerance', 'anxiety']
        },
        'Kidney Disease': {
            'tests': ['creatinine', 'bun', 'egfr', 'gfr', 'urea', 'uric acid'],
            'low_tests': ['egfr', 'gfr'],
            'high_tests': ['creatinine', 'bun', 'urea', 'uric acid'],
            'keywords': ['kidney disease', 'renal insufficiency', 'nephropathy', 'proteinuria']
        },
        'Liver Disease': {
            'tests': ['alt', 'ast', 'alp', 'bilirubin', 'ggt', 'albumin', 'total protein'],
            'low_tests': ['albumin', 'total protein'],
            'high_tests': ['alt', 'ast', 'alp', 'bilirubin', 'ggt'],
            'keywords': ['liver disease', 'hepatitis', 'cirrhosis', 'fatty liver', 'jaundice']
        },
        'Infection': {
            'tests': ['wbc', 'crp', 'esr', 'procalcitonin'],
            'low_tests': [],
            'high_tests': ['wbc', 'crp', 'esr', 'procalcitonin'],
            'keywords': ['infection', 'inflammation', 'fever', 'leukocytosis']
        },
        'Vitamin D Deficiency': {
            'tests': ['vitamin d', '25-oh vitamin d', 'vitamin d3', 'calciferol'],
            'low_tests': ['vitamin d', '25-oh vitamin d', 'vitamin d3', 'calciferol'],
            'high_tests': [],
            'keywords': ['vitamin d deficiency', 'vitamin d insufficient', 'bone pain', 'weakness']
        },
        'Vitamin B12 Deficiency': {
            'tests': ['vitamin b12', 'cobalamin', 'mma', 'homocysteine'],
            'low_tests': ['vitamin b12', 'cobalamin'],
            'high_tests': ['mma', 'homocysteine'],
            'keywords': ['vitamin b12 deficiency', 'pernicious anemia', 'neuropathy', 'macrocytic anemia']
        },
        'Electrolyte Imbalance': {
            'tests': ['sodium', 'potassium', 'chloride', 'calcium', 'magnesium', 'phosphorus'],
            'low_tests': ['sodium', 'potassium', 'chloride', 'calcium', 'magnesium', 'phosphorus'],
            'high_tests': ['sodium', 'potassium', 'chloride', 'calcium', 'magnesium', 'phosphorus'],
            'keywords': ['electrolyte imbalance', 'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia']
        },
        'Thrombocytopenia': {
            'tests': ['platelets', 'platelet count'],
            'low_tests': ['platelets', 'platelet count'],
            'high_tests': [],
            'keywords': ['thrombocytopenia', 'low platelet count', 'bruising', 'bleeding']
        }
    }
    
    # List to store identified health issues
    health_issues = []
    
    # Check for explicit diagnosis statements
    diagnosis_patterns = [
        r'(?:diagnosis|assessment|conclusion|impression)(?:\s*(?:of|:))?\s*([^.;]*)',
        r'(?:patient|subject)(?:\s+(?:has|with|diagnosed\s+with))(?:\s+an?)?(?:\s+(?:acute|chronic))?\s+([^.;]*)',
        r'(?:findings\s+(?:consistent|suggestive|compatible)\s+with)(?:\s+an?)?(?:\s+(?:acute|chronic))?\s+([^.;]*)'
    ]
    
    for pattern in diagnosis_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            diagnosis = match.group(1).strip()
            if diagnosis and len(diagnosis) > 3 and diagnosis.lower() not in ['none', 'normal', 'unremarkable']:
                health_issues.append({
                    'condition': diagnosis.strip(),
                    'confidence': 'high',
                    'evidence': 'Explicit diagnosis in report',
                    'source': 'diagnosis'
                })
    
    # Check abnormal values against known conditions
    for value in abnormal_values:
        test_name = value['test'].lower()
        direction = value['direction']
        
        for condition, condition_data in health_conditions.items():
            # Check if this test is relevant for this condition
            if any(test.lower() in test_name.lower() for test in condition_data['tests']):
                # Check if the direction matches what's expected for this condition
                matches_direction = False
                
                if direction == 'high' and any(test.lower() in test_name.lower() for test in condition_data['high_tests']):
                    matches_direction = True
                elif direction == 'low' and any(test.lower() in test_name.lower() for test in condition_data['low_tests']):
                    matches_direction = True
                
                if matches_direction:
                    # Check if we've already added this condition
                    if not any(issue['condition'] == condition for issue in health_issues):
                        health_issues.append({
                            'condition': condition,
                            'confidence': 'medium',
                            'evidence': f"Abnormal {test_name}: {value['value']} ({direction})",
                            'source': 'lab_value'
                        })
    
    # Check for condition keywords in the text
    for condition, condition_data in health_conditions.items():
        for keyword in condition_data['keywords']:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                # Check if we've already added this condition
                if not any(issue['condition'] == condition for issue in health_issues):
                    health_issues.append({
                        'condition': condition,
                        'confidence': 'low',
                        'evidence': f"Keyword '{keyword}' found in report",
                        'source': 'keyword'
                    })
    
    # Sort by confidence: high > medium > low
    confidence_order = {'high': 0, 'medium': 1, 'low': 2}
    health_issues.sort(key=lambda x: confidence_order[x['confidence']])
    
    return health_issues

def generate_recommendations(health_issues):
    """
    Generate recommendations based on identified health issues
    
    Args:
        health_issues (list): List of identified health issues
        
    Returns:
        list: Recommendations for follow-up and management
    """
    recommendations = []
    
    # General recommendation for any abnormal findings
    if health_issues:
        recommendations.append("Discuss these results with your healthcare provider.")
    
    # Specific recommendations based on conditions
    condition_recommendations = {
        'Anemia': [
            "Consider iron supplementation if iron deficiency is confirmed.",
            "Dietary changes to include iron-rich foods.",
            "Follow-up complete blood count in 3 months."
        ],
        'Diabetes': [
            "Monitor blood glucose levels regularly.",
            "Consider diabetes management plan including diet, exercise, and possibly medication.",
            "Follow-up HbA1c test in 3 months."
        ],
        'Hyperlipidemia': [
            "Dietary changes to reduce saturated fat and cholesterol intake.",
            "Regular exercise regimen.",
            "Consider statin therapy if lifestyle changes insufficient.",
            "Follow-up lipid panel in 6 months."
        ],
        'Hypertension': [
            "Regular blood pressure monitoring.",
            "Reduce sodium intake.",
            "Regular exercise and weight management.",
            "Consider antihypertensive medication if lifestyle changes insufficient."
        ],
        'Hypothyroidism': [
            "Consider thyroid hormone replacement therapy.",
            "Follow-up thyroid function tests in 6-8 weeks."
        ],
        'Hyperthyroidism': [
            "Consider antithyroid medication, radioactive iodine, or surgery.",
            "Follow-up thyroid function tests in 4-6 weeks."
        ],
        'Kidney Disease': [
            "Nephrology consultation recommended.",
            "Monitor blood pressure and kidney function regularly.",
            "Dietary modifications to reduce sodium, protein, and potassium."
        ],
        'Liver Disease': [
            "Hepatology consultation recommended.",
            "Avoid alcohol and hepatotoxic medications.",
            "Follow-up liver function tests in 4-6 weeks."
        ],
        'Infection': [
            "Identify source of infection.",
            "Consider appropriate antimicrobial therapy.",
            "Follow-up laboratory tests to monitor response."
        ],
        'Vitamin D Deficiency': [
            "Vitamin D supplementation.",
            "Sun exposure for 15-30 minutes several times a week.",
            "Follow-up vitamin D level in 3 months."
        ],
        'Vitamin B12 Deficiency': [
            "Vitamin B12 supplementation (oral or injectable).",
            "Dietary changes to include B12-rich foods.",
            "Follow-up B12 level in 3 months."
        ],
        'Electrolyte Imbalance': [
            "Identify and treat underlying cause.",
            "Electrolyte replacement as needed.",
            "Follow-up electrolyte panel in 1-2 weeks."
        ],
        'Thrombocytopenia': [
            "Hematology consultation recommended.",
            "Avoid medications that affect platelets (e.g., aspirin, NSAIDs).",
            "Follow-up complete blood count in 1-2 weeks."
        ]
    }
    
    # Add condition-specific recommendations
    for issue in health_issues:
        condition = issue['condition']
        # Check if it's one of our predefined conditions
        if condition in condition_recommendations:
            # Add the first recommendation for this condition if we haven't already
            first_rec = condition_recommendations[condition][0]
            if first_rec not in recommendations:
                recommendations.append(first_rec)
            
            # For high confidence issues, add more detailed recommendations
            if issue['confidence'] == 'high':
                for rec in condition_recommendations[condition][1:]:
                    if rec not in recommendations:
                        recommendations.append(rec)
        else:
            # For custom conditions, add a generic recommendation
            rec = f"Follow up with your healthcare provider about {condition}."
            if rec not in recommendations:
                recommendations.append(rec)
    
    # If no specific issues were found
    if not recommendations:
        recommendations.append("No specific health issues identified. Continue routine health maintenance.")
    
    return recommendations

def assess_risk_level(abnormal_values, health_issues):
    """
    Assess overall risk level based on abnormal values and identified health issues
    
    Args:
        abnormal_values (list): List of abnormal test values
        health_issues (list): List of identified health issues
        
    Returns:
        str: Risk level assessment ('Low', 'Moderate', or 'High')
    """
    # Define high-risk conditions
    high_risk_conditions = [
        'diabetes', 'hypertension', 'kidney disease', 'liver disease', 
        'severe anemia', 'infection', 'electrolyte imbalance', 'heart disease',
        'stroke', 'cancer', 'leukemia', 'thrombocytopenia', 'pulmonary embolism',
        'sepsis', 'acute kidney injury', 'myocardial infarction'
    ]
    
    # Check for critical abnormal values
    critical_abnormal = False
    for value in abnormal_values:
        test_name = value['test'].lower()
        value_num = value['value']
        direction = value['direction']
        
        # Define critical thresholds for common tests
        if isinstance(value_num, (int, float)):
            if (test_name == 'hemoglobin' and value_num < 7) or \
               (test_name == 'glucose' and value_num > 300) or \
               (test_name == 'potassium' and (value_num < 2.5 or value_num > 6.5)) or \
               (test_name == 'sodium' and (value_num < 125 or value_num > 155)) or \
               (test_name == 'platelets' and value_num < 50) or \
               (test_name == 'wbc' and value_num > 20) or \
               ((test_name == 'alt' or test_name == 'ast') and value_num > 1000):
                critical_abnormal = True
    
    # Count high-confidence health issues
    high_confidence_issues = [issue for issue in health_issues if issue['confidence'] == 'high']
    
    # Count high-risk conditions
    high_risk_count = 0
    for issue in health_issues:
        condition = issue['condition'].lower()
        if any(risk in condition for risk in high_risk_conditions):
            high_risk_count += 1
    
    # Determine risk level
    if critical_abnormal or high_risk_count > 0 or len(high_confidence_issues) >= 3:
        return 'High'
    elif len(high_confidence_issues) > 0 or len(abnormal_values) > 5:
        return 'Moderate'
    else:
        return 'Low'

def generate_report_summary(text):
    """Generate a summary of the medical report text with highlights of abnormalities."""
    import re
    
    # Analyze the report
    analysis_results = analyze_medical_report(text)
    
    # Create a summary with sections
    summary = "# Medical Report Analysis\n\n"
    
    # Add risk level assessment
    risk_level = analysis_results['risk_level']
    summary += f"### Key Findings\n"
    summary += f"- **Risk Level**: {risk_level}\n"
    
    # Add health issues section
    summary += f"- **Health Issues Identified**:\n"
    if analysis_results['health_issues']:
        for issue in analysis_results['health_issues']:
            summary += f"  - {issue['condition']}: {issue['evidence']}\n"
    else:
        summary += "  - No significant health issues identified\n"
    
    # Add abnormal values section
    summary += f"- **Abnormal Test Results**:\n"
    if analysis_results['abnormal_values']:
        for value in analysis_results['abnormal_values']:
            direction_indicator = {
                'high': '↑ HIGH',
                'low': '↓ LOW',
                'abnormal': '⚠️ ABNORMAL'
            }.get(value['direction'], '')
            units = f" {value['units']}" if value['units'] else ""
            summary += f"  - **{value['test']}:** {value['value']}{units} ({direction_indicator})\n"
    else:
        summary += "  - No abnormal test results identified\n"
    
    # Add areas for improvement and recommendations
    summary += f"\n### Areas for Improvement\n"
    
    # Recommendations based on health issues
    if analysis_results['health_issues']:
        for issue in analysis_results['health_issues']:
            condition = issue['condition']
            if condition == 'Anemia':
                summary += f"- **Anemia**:\n"
                summary += "  - Consider iron supplementation if iron deficiency is confirmed.\n"
                summary += "  - Include iron-rich foods in the diet (e.g., spinach, red meat).\n"
                summary += "  - Follow-up complete blood count in 3 months.\n"
            elif condition == 'Hypertension':
                summary += f"- **Hypertension**:\n"
                summary += "  - Regular blood pressure monitoring.\n"
                summary += "  - Reduce sodium intake in the diet.\n"
                summary += "  - Implement a regular exercise regimen.\n"
                summary += "  - Consider antihypertensive medication if lifestyle changes are insufficient.\n"
    
    # General recommendations
    summary += f"\n### General Recommendations\n"
    summary += "- Discuss these results with your healthcare provider for personalized advice and management.\n"
    summary += "- Maintain a balanced diet and regular physical activity to support overall health.\n"
    
    return summary


def extract_medical_terms(text):
    # Common medical abbreviations and terms with definitions
    common_medical_terms = {
        "CBC": "Complete Blood Count - A blood test used to evaluate overall health",
        "WBC": "White Blood Cell - Cells that help fight infection",
        "RBC": "Red Blood Cell - Cells that carry oxygen",
        "HGB": "Hemoglobin - Protein in red blood cells that carries oxygen",
        "HCT": "Hematocrit - Percentage of blood volume that is red blood cells",
        "PLT": "Platelets - Blood cells that help with clotting",
        "BUN": "Blood Urea Nitrogen - Waste product filtered by kidneys",
        "Cr": "Creatinine - Waste product filtered by kidneys",
        "Na": "Sodium - An electrolyte that helps maintain fluid balance",
        "K": "Potassium - An electrolyte essential for heart function",
        "Cl": "Chloride - An electrolyte that works with sodium",
        "CO2": "Carbon Dioxide - A measure of carbon dioxide in blood",
        "Ca": "Calcium - Essential for bone health and nerve function",
        "Glu": "Glucose - Blood sugar level",
        "A1C": "Hemoglobin A1C - Average blood glucose over past 3 months",
        "ALT": "Alanine Aminotransferase - Liver enzyme",
        "AST": "Aspartate Aminotransferase - Liver enzyme",
        "ALP": "Alkaline Phosphatase - Enzyme found in liver and bones",
        "GGT": "Gamma-Glutamyl Transferase - Liver enzyme",
        "Bili": "Bilirubin - Product of red blood cell breakdown",
        "Chol": "Cholesterol - Fat-like substance in blood",
        "HDL": "High-Density Lipoprotein - 'Good' cholesterol",
        "LDL": "Low-Density Lipoprotein - 'Bad' cholesterol",
        "TG": "Triglycerides - Type of fat in blood",
        "TSH": "Thyroid Stimulating Hormone - Hormone that regulates thyroid",
        "FT4": "Free Thyroxine - Thyroid hormone",
        "ESR": "Erythrocyte Sedimentation Rate - Indicator of inflammation",
        "CRP": "C-Reactive Protein - Indicator of inflammation",
        "RF": "Rheumatoid Factor - Antibody found in rheumatoid arthritis",
        "ANA": "Antinuclear Antibody - Antibody that attacks cellular components",
        "ECG": "Electrocardiogram - Test that checks heart's electrical activity",
        "EKG": "Electrocardiogram - Test that checks heart's electrical activity",
        "BP": "Blood Pressure - Force of blood against artery walls",
        "HR": "Heart Rate - Number of heartbeats per minute",
        "RR": "Respiratory Rate - Number of breaths per minute",
        "SpO2": "Oxygen Saturation - Amount of oxygen in blood",
        "BMI": "Body Mass Index - Measure of body fat based on height and weight",
        "UTI": "Urinary Tract Infection - Infection in any part of urinary system",
        "URI": "Upper Respiratory Infection - Infection of nose, throat, airways",
        "PTT": "Partial Thromboplastin Time - Blood clotting test",
        "PT": "Prothrombin Time - Blood clotting test",
        "INR": "International Normalized Ratio - Blood clotting test",
        "MRI": "Magnetic Resonance Imaging - Imaging test using magnetic fields",
        "CT": "Computed Tomography - X-ray imaging test",
        "US": "Ultrasound - Imaging test using sound waves",
        "PET": "Positron Emission Tomography - Nuclear medicine imaging test",
        "DM": "Diabetes Mellitus - Condition affecting blood sugar control",
        "HTN": "Hypertension - High blood pressure",
        "CAD": "Coronary Artery Disease - Narrowing of coronary arteries",
        "CHF": "Congestive Heart Failure - Heart can't pump efficiently",
        "COPD": "Chronic Obstructive Pulmonary Disease - Lung disease",
        "DVT": "Deep Vein Thrombosis - Blood clot in a deep vein",
        "PE": "Pulmonary Embolism - Blood clot in lungs",
        "RA": "Rheumatoid Arthritis - Autoimmune joint inflammation",
        "SLE": "Systemic Lupus Erythematosus - Autoimmune disease",
        "IBD": "Inflammatory Bowel Disease - Chronic gut inflammation",
        "CKD": "Chronic Kidney Disease - Gradual loss of kidney function",
        "ESRD": "End-Stage Renal Disease - Kidney failure requiring dialysis",
        "GERD": "Gastroesophageal Reflux Disease - Chronic acid reflux",
        "BPH": "Benign Prostatic Hyperplasia - Enlarged prostate",
        "OA": "Osteoarthritis - Joint degeneration",
        "OSA": "Obstructive Sleep Apnea - Breathing stops during sleep",
        "FEV1": "Forced Expiratory Volume in 1 Second - Lung function test",
        "FVC": "Forced Vital Capacity - Lung function test",
        "DLCO": "Diffusing Capacity of the Lungs for Carbon Monoxide - Lung test",
        "GFR": "Glomerular Filtration Rate - Kidney function test",
    }
    
    found_terms = {}
    
    # Look for each term in the text
    for term, definition in common_medical_terms.items():
        # Create regex pattern that matches the term as a whole word
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_terms[term] = definition
    
    return found_terms

if __name__ == '__main__':
    app.run(debug=True)