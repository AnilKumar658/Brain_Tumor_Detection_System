import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

from flask import Flask, request, render_template, redirect, url_for,send_file
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import matplotlib.patches as patches

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="gJ4rexpzJnj6QkwPRT5q"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def classify_tumor_stage(size_cm):
    if size_cm <= 2:
        return "T1"
    elif 2 < size_cm <= 5:
        return "T2"
    elif 5 < size_cm:
        return "T3"
    else:
        return "T4"

def get_brain_tumor_grade(stage):
    brain_tumor_grades = {
        "T1": {
            "Name": "Grade I",
            "Description": "Benign tumors, slow-growing with well-defined edges."
        },
        "T2": {
            "Name": "Grade II",
            "Description": "Low-grade tumors that can be infiltrative and may recur."
        },
        "T3": {
            "Name": "Grade III",
            "Description": "Malignant tumors that are actively growing."
        },
        "T4": {
            "Name": "Grade IV",
            "Description": "Highly malignant tumors with a poor prognosis."
        }
    }
    return brain_tumor_grades.get(stage, {"Name": "Unknown", "Description": "No description available."})

def process_image(image_path):
    DPI = 300
    result = CLIENT.infer(image_path, model_id="brain-tumor-detector-dqbqk/3")
    image = Image.open(image_path)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    tumor_images = []
    tumor_image_paths = []
    tumor_details = []
    
    for i, prediction in enumerate(result['predictions']):
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width, 
            height, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(x - width / 2, y - height / 2 - 10, f"{prediction['class']}", color='red', fontsize=20)

        tumor_size_cm = (width / DPI) * (height / DPI) * (2.54 * 4)
        tumor_stage = classify_tumor_stage(tumor_size_cm)
        grade_details = get_brain_tumor_grade(tumor_stage)

        tumor_details.append({
            "size_cm": tumor_size_cm,
            "stage": tumor_stage,
            "grade_name": grade_details['Name'],
            "description": grade_details['Description']
        })

        left = int(x - width / 2)
        upper = int(y - height / 2)
        right = int(x + width / 2)
        lower = int(y + height / 2)
        tumor_image = image.crop((left, upper, right, lower))
        
        tumor_image = tumor_image.filter(ImageFilter.SHARPEN)
        tumor_image_gray = tumor_image.convert('L')
        threshold = 128
        tumor_image_binary = tumor_image_gray.point(lambda p: 0 if p > threshold else 255, mode='1')
        
        cropped_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'cropped_{i}_{os.path.basename(image_path)}')
        tumor_image_binary.save(cropped_image_path)
        tumor_image_paths.append(cropped_image_path)
    ax.axis('off')
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    fig.savefig(processed_image_path)
    plt.close(fig)

    processed_image_url = url_for('static', filename='uploads/processed_' + os.path.basename(image_path))
    
    tumor_image_urls = []
    for i, path in enumerate(tumor_image_paths):
        # Use the relative path from the 'static/uploads' folder
        tumor_image_url = url_for('static', filename='uploads/cropped_{}_{}'.format(i, os.path.basename(image_path)))
        tumor_image_urls.append(tumor_image_url)
    return processed_image_url, tumor_image_urls, tumor_details
 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('result', filename=filename))
    return render_template('index.html')

@app.route('/result/<filename>', methods=['GET'])

def result(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_image_path, tumor_image_paths, tumor_details = process_image(file_path)
    
    # Convert tumor image paths to URLs for rendering
    tumor_image_urls = []
    for path in tumor_image_paths:
        # Correctly generate the URL for the cropped images
        tumor_image_url = url_for('static', filename=f'uploads/{os.path.basename(path)}')
        tumor_image_urls.append(tumor_image_url)
    
    print(tumor_image_urls)  # This will show the correct URLs in the console
    return render_template('result.html', filename=filename, processed_image_path=processed_image_path, tumor_images=tumor_image_urls, tumor_details=tumor_details)

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/view-pdf')
def view_pdf():
    # Path to your PDF file
    pdf_path = r'C:\Users\anilk\Mini Project\segmentation\model_evaluation_report1.pdf'
    return send_file(pdf_path, mimetype='application/pdf', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
