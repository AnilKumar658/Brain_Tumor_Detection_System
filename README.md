# Brain Tumor Detection System 🧠💻

This repository contains the implementation of a Brain Tumor Detection System that uses deep learning and image processing techniques. The system leverages a hybrid approach combining Convolutional Neural Networks (CNNs) and Transformers to classify and evaluate brain tumors based on Magnetic Resonance Imaging (MRI).

# Features ✨

Deep Learning Models: Pretrained models (stored as .h5 files) for accurate tumor detection.

Interactive Web Application: A Streamlit-based app for user interaction.

Model Evaluation: Detailed metrics and evaluation reports.

Hybrid Architecture: Combines CNNs and Transformers for enhanced performance.

# File Structure 📁

app.py: Main script to run the web application.

barin_tumor.h5 and Brain-Tumor-Classification.h5: Pretrained deep learning models.

brain_tumor.pkl: Serialized model or pipeline file.

model metrics.ipynb: Jupyter notebook for analyzing model performance metrics.

model_evaluation_report.pdf and model_evaluation_report1.pdf: Model evaluation reports in PDF format.

project.ipynb: Jupyter notebook containing the main project development workflow.

static/: Directory containing static assets like CSS, images, and JavaScript files.

templates/: Directory containing HTML templates for the web application.

__pycache__/: Directory containing compiled Python files.

# Installation 🛠️

Clone the repository:

git clone https://github.com/anilkumar658/brain-tumor-detection.git
cd brain-tumor-detection/segmentation

Create a virtual environment:

python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Download the pretrained models and place them in the root directory.

# Usage 🚀

To run the web application:

python app.py

The application will start, and you can interact with it via your browser at http://localhost:8501/.

# Model Evaluation 📊

Model performance is documented in model_evaluation_report.pdf and model_evaluation_report1.pdf.

Metrics and visualizations can be explored in model metrics.ipynb.

# Requirements 📋

Ensure you have the following installed:

Python 3.8+
TensorFlow
Streamlit
Contribution 🤝
Feel free to submit issues or pull requests to improve the system.

# License 📜

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments 🙏

This project is a part of the final year research work titled "Brain Tumor Detection System From Magnetic Resonance Imaging Using Deep Learning Techniques And Image Processing."
