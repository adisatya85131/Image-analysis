# Image-analysis
# AI Person Classifier

A Flask-based AI web application that uses a TensorFlow/Keras model to identify people in real time using a webcam. The application captures images from the camera, processes them with a trained model, and displays prediction results with confidence scores.

## Features

* Real-time webcam access
* AI-powered image classification
* TensorFlow/Keras model integration
* Responsive modern UI
* Confidence score display
* Health check endpoint
* Mobile-friendly design
* Teachable Machine compatible preprocessing

## Project Structure

```bash
project/
│── app.py
│── keras_model.h5
│── labels.txt
│── requirements.txt
```

## Requirements

```txt
Flask==2.3.3
tensorflow==2.13.0
Pillow==10.0.1
numpy==1.24.3
Werkzeug==2.3.7
```

## Installation

### Clone the Project

```bash
git clone <repository-url>
cd project-folder
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Model Setup

Place your trained model file in the project directory:

```bash
keras_model.h5
```

Update class names in `app.py`:

```python
class_names = ["Chaitu", "Shyam"]
```

Update labels in `labels.txt`:

```txt
0 Class 1
1 Class 2
```

## Running the Application

```bash
python app.py
```

The application will start at:

```bash
http://localhost:5000
```

## How It Works

1. User opens the web app
2. Browser requests camera permission
3. User captures image
4. Image is sent to Flask backend
5. TensorFlow model processes the image
6. Prediction result and confidence score are displayed

## API Endpoints

### Home Page

```http
GET /
```

Loads the main UI.

### Predict Endpoint

```http
POST /predict
```

Accepts image input and returns prediction results.

#### Example Response

```json
{
  "class": 1,
  "class_name": "Chaitu",
  "confidence": 0.98
}
```

### Health Check

```http
GET /health
```

#### Example Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Image Preprocessing

The application preprocesses images before prediction:

* Resize to 224x224
* Convert to RGB
* Normalize pixel values to range [-1, 1]

```python
image_array = (image_array / 127.5) - 1
```

## Technologies Used

* Python
* Flask
* TensorFlow/Keras
* NumPy
* Pillow
* HTML/CSS/JavaScript

## Future Improvements

* Face detection support
* Multi-person recognition
* Real-time live prediction
* Authentication system
* Database integration
* Model upload dashboard

## License

This project is open-source and intended for educational and personal use.
