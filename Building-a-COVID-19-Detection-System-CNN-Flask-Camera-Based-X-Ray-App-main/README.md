# COVID-19 Detection System using CNN, Flask, and Camera-Based X-Ray App

## Overview

This project is a web-based application for detecting COVID-19 from chest X-ray images using a Convolutional Neural Network (CNN) model. The system allows users to upload X-ray images or capture them directly using their device's camera, and provides real-time predictions with confidence scores. The application is built using Flask for the backend, HTML/CSS/JavaScript for the frontend, and TensorFlow/Keras for the machine learning model.

## Features

- **Image Upload**: Users can upload chest X-ray images in various formats (e.g., JPG, PNG).
- **Camera Capture**: Real-time camera access to capture X-ray images directly from the device.
- **CNN Prediction**: Utilizes a pre-trained CNN model to classify images into three categories:
  - COVID-19
  - Viral Pneumonia
  - Normal
- **Confidence Score**: Displays the prediction confidence percentage for transparency.
- **Responsive Design**: Mobile-friendly interface that works on desktop and mobile devices.
- **Secure File Handling**: Uses secure filename handling to prevent security vulnerabilities.

## Technology Stack

### Backend

- **Flask**: Lightweight web framework for Python.
- **TensorFlow/Keras**: Deep learning framework for loading and running the CNN model.
- **OpenCV**: Image processing library for resizing and preprocessing images.
- **NumPy**: Numerical computing library for array operations.
- **Pickle**: For loading the label encoder.

### Frontend

- **HTML5**: Structure of the web pages.
- **CSS3**: Styling for responsive and attractive UI.
- **JavaScript**: Client-side scripting for camera functionality and form handling.
- **Canvas API**: For capturing images from the camera stream.

### Machine Learning

- **CNN Model**: Custom Convolutional Neural Network trained on chest X-ray datasets.
- **Label Encoder**: Encodes categorical labels for prediction output.

## Project Structure

```
Building-a-COVID-19-Detection-System-CNN-Flask-Camera-Based-X-Ray-App-main/
│
├── app.py                          # Main Flask application
├── building-a-covid-19-detection-system-using-cnn-dl.ipynb  # Jupyter notebook for model training
├── index.html                      # Home page with upload form
├── camera.html                     # Camera capture page
├── result.html                     # Results display page
├── models/                         # Directory for model files
│   ├── CNN_Covid19_Xray_Version.h5 # Trained CNN model
│   └── Label_encoder.pkl           # Label encoder
├── uploads/                        # Directory for uploaded images (created dynamically)
└── README.md                       # This file
```

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- A web browser with camera support (for camera functionality)

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/deekkssss/covid-detection.git
   cd covid-detection
   ```

2. **Create a Virtual Environment** (Recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install flask tensorflow opencv-python numpy pickle-mixin
   ```

4. **Prepare Model Files**

   - Place your trained CNN model file (`CNN_Covid19_Xray_Version.h5`) in the `models/` directory.
   - Place the label encoder file (`Label_encoder.pkl`) in the `models/` directory.
   - If these files are not present, you may need to train the model using the provided Jupyter notebook.

5. **Run the Application**

   ```bash
   python app.py
   ```

6. **Access the Application**
   - Open your web browser and navigate to `http://localhost:5000`
   - The application will be running in debug mode.

## Usage

### Uploading an Image

1. On the home page, click "Choose File" to select a chest X-ray image.
2. Click "Upload Image" to submit the file.
3. View the prediction results on the results page.

### Using the Camera

1. Click "Open Camera" on the home page.
2. Grant camera permissions when prompted.
3. Position the X-ray image in front of the camera.
4. Click "Capture Image" to take a photo.
5. The image will be automatically processed and results displayed.

### Interpreting Results

- **Predicted Status**: Shows the classification (COVID-19, Viral Pneumonia, or Normal).
- **Confidence Score**: Indicates the model's confidence in the prediction (higher is better).
- Use this information as a supplementary tool, not a definitive diagnosis.

## Model Training

The CNN model is trained using the Jupyter notebook `building-a-covid-19-detection-system-using-cnn-dl.ipynb`. This notebook includes:

- Data preprocessing and augmentation
- CNN architecture definition
- Model training and validation
- Performance evaluation metrics
- Model saving and label encoder creation

To retrain or modify the model:

1. Open the notebook in Jupyter Lab or Jupyter Notebook.
2. Ensure you have the required datasets (chest X-ray images for COVID-19, pneumonia, and normal cases).
3. Run the cells sequentially to train a new model.
4. Update the model and encoder file paths in `app.py` if necessary.

## API Endpoints

The Flask application exposes the following endpoints:

- `GET /`: Home page with upload form.
- `POST /upload`: Handles image upload and processing.
- `GET /camera`: Camera capture page.
- `GET /uploads/<filename>`: Serves uploaded images.

## Security Considerations

- File uploads are limited to image formats only.
- Filenames are secured using `werkzeug.utils.secure_filename`.
- The application runs in debug mode; disable for production use.
- Consider implementing authentication for production deployments.

## Performance and Limitations

- **Accuracy**: Model performance depends on the quality and diversity of training data.
- **Processing Time**: Image processing and prediction may take a few seconds.
- **Device Compatibility**: Camera functionality requires modern browsers with MediaDevices API support.
- **False Positives/Negatives**: This is an AI-assisted tool; always consult medical professionals for diagnosis.
