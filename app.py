from flask import Flask, render_template_string, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and class names
model = None
# CUSTOMIZE THESE NAMES TO MATCH YOUR TRAINED CLASSES
class_names = ["Chaitu", "Shyam"]  # Change these to the actual names of the two people

def load_model_and_setup():
    """Load the Keras model and setup"""
    global model
    try:
        # Load your trained model
        model_path = 'keras_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Model loaded successfully from {model_path}!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info(f"Class names: {class_names}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for prediction - Teachable Machine format"""
    try:
        # Teachable Machine uses 224x224 input size
        target_size = (224, 224)
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Teachable Machine normalization: (pixel_value / 127.5) - 1
        # This scales pixel values from [0, 255] to [-1, 1]
        image_array = (image_array / 127.5) - 1
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image preprocessed - Shape: {image_array.shape}, Min: {image_array.min():.3f}, Max: {image_array.max():.3f}")
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Person Classifier - AI Recognition</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}

            .container {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                padding: 40px;
                max-width: 900px;
                width: 100%;
                text-align: center;
            }}

            h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .subtitle {{
                color: #666;
                margin-bottom: 20px;
                font-size: 1.2em;
            }}

            .class-info {{
                background: linear-gradient(45deg, #f093fb, #f5576c);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 30px;
                font-weight: 600;
            }}

            .camera-container {{
                position: relative;
                display: inline-block;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                margin-bottom: 30px;
            }}

            #video {{
                width: 500px;
                height: 375px;
                object-fit: cover;
                display: block;
            }}

            #canvas {{
                display: none;
            }}

            .controls {{
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}

            button {{
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                min-width: 150px;
            }}

            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }}

            button:disabled {{
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }}

            .result-container {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border-radius: 15px;
                padding: 30px;
                margin-top: 20px;
                color: white;
                font-size: 1.5em;
                font-weight: 600;
                min-height: 120px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
            }}

            .prediction-text {{
                font-size: 2.2em;
                margin-bottom: 15px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}

            .confidence {{
                font-size: 1.3em;
                opacity: 0.9;
                margin-bottom: 10px;
            }}

            .details {{
                font-size: 0.9em;
                opacity: 0.8;
                border-top: 1px solid rgba(255,255,255,0.3);
                padding-top: 15px;
                margin-top: 15px;
            }}

            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
                margin-right: 10px;
            }}

            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}

            .status {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 10px;
                font-weight: 500;
            }}

            .status.success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}

            .status.error {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}

            .status.info {{
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }}

            @media (max-width: 700px) {{
                .container {{
                    padding: 20px;
                }}
                
                #video {{
                    width: 350px;
                    height: 262px;
                }}
                
                h1 {{
                    font-size: 2em;
                }}
                
                .controls {{
                    flex-direction: column;
                    align-items: center;
                }}
                
                button {{
                    width: 200px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 AI Person Classifier</h1>
            <p class="subtitle">Advanced facial recognition system</p>
            
            <div class="class-info">
                🔍 Trained to recognize: <strong>{class_names[0]}</strong> (Class 1) vs <strong>{class_names[1]}</strong> (Class 2)
            </div>
            
            <div id="status" class="status info">
                Click "Start Camera" to begin person identification
            </div>

            <div class="camera-container">
                <video id="video" autoplay muted playsinline></video>
                <canvas id="canvas"></canvas>
            </div>

            <div class="controls">
                <button id="startBtn" onclick="startCamera()">📹 Start Camera</button>
                <button id="captureBtn" onclick="captureAndPredict()" disabled>🔍 Identify Person</button>
                <button id="stopBtn" onclick="stopCamera()" disabled>⏹️ Stop Camera</button>
            </div>

            <div id="result" class="result-container">
                <div>🚀 Ready to identify people! Start your camera first.</div>
            </div>
        </div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let ctx = canvas.getContext('2d');
            let stream = null;
            let isProcessing = false;

            const statusDiv = document.getElementById('status');
            const resultDiv = document.getElementById('result');
            const startBtn = document.getElementById('startBtn');
            const captureBtn = document.getElementById('captureBtn');
            const stopBtn = document.getElementById('stopBtn');

            function updateStatus(message, type = 'info') {{
                statusDiv.textContent = message;
                statusDiv.className = `status ${{type}}`;
            }}

            async function startCamera() {{
                try {{
                    updateStatus('🎥 Requesting camera access...', 'info');
                    
                    const constraints = {{
                        video: {{
                            width: {{ ideal: 1280 }},
                            height: {{ ideal: 720 }},
                            facingMode: 'user'
                        }}
                    }};

                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = stream;
                    
                    video.onloadedmetadata = () => {{
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        startBtn.disabled = true;
                        captureBtn.disabled = false;
                        stopBtn.disabled = false;
                        
                        updateStatus('✅ Camera is ready! You can now identify people.', 'success');
                        resultDiv.innerHTML = '<div>📹 Camera active - Ready to identify people!</div>';
                    }};

                }} catch (err) {{
                    console.error('Error accessing camera:', err);
                    let errorMessage = '❌ Error accessing camera. ';
                    
                    if (err.name === 'NotAllowedError') {{
                        errorMessage += 'Please allow camera access and refresh the page.';
                    }} else if (err.name === 'NotFoundError') {{
                        errorMessage += 'No camera found on this device.';
                    }} else {{
                        errorMessage += err.message || 'Unknown error occurred.';
                    }}
                    
                    updateStatus(errorMessage, 'error');
                    resultDiv.innerHTML = '<div>❌ Camera access failed</div>';
                }}
            }}

            function stopCamera() {{
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }}
                
                video.srcObject = null;
                
                startBtn.disabled = false;
                captureBtn.disabled = true;
                stopBtn.disabled = true;
                
                updateStatus('📹 Camera stopped. Click "Start Camera" to begin again.', 'info');
                resultDiv.innerHTML = '<div>📹 Camera stopped - Click Start Camera to begin identification</div>';
            }}

            async function captureAndPredict() {{
                if (isProcessing) return;
                
                isProcessing = true;
                captureBtn.disabled = true;
                
                try {{
                    updateStatus('🔍 Capturing and analyzing face...', 'info');
                    resultDiv.innerHTML = '<div><span class="loading"></span>🤖 AI is analyzing the person...</div>';

                    // Capture current frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert canvas to blob with high quality
                    const blob = await new Promise(resolve => {{
                        canvas.toBlob(resolve, 'image/jpeg', 0.95);
                    }});

                    // Create form data
                    const formData = new FormData();
                    formData.append('image', blob, 'capture.jpg');

                    // Send to Flask backend
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }});

                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}

                    const result = await response.json();
                    
                    if (result.error) {{
                        throw new Error(result.error);
                    }}
                    
                    displayResult(result);
                    updateStatus('✅ Person identification complete!', 'success');

                }} catch (error) {{
                    console.error('Error during prediction:', error);
                    updateStatus('❌ Error during identification. Please try again.', 'error');
                    resultDiv.innerHTML = '<div>❌ Identification failed - Please try again</div>';
                }} finally {{
                    isProcessing = false;
                    captureBtn.disabled = false;
                }}
            }}

            function displayResult(result) {{
                const className = result.class_name;
                const confidence = Math.round(result.confidence * 100);
                const emoji = result.class === 1 ? "👤" : "👥";
                
                // Get both class confidences for better display
                const class1Conf = Math.round(result.all_classes.class_1.confidence * 100);
                const class2Conf = Math.round(result.all_classes.class_2.confidence * 100);
                
                // Determine confidence level
                let confidenceLevel = "";
                if (confidence >= 90) confidenceLevel = "🎯 Very High";
                else if (confidence >= 75) confidenceLevel = "✅ High";
                else if (confidence >= 60) confidenceLevel = "⚠️ Medium";
                else confidenceLevel = "❓ Low";
                
                resultDiv.innerHTML = `
                    <div>
                        <div class="prediction-text">${{emoji}} ${{className}}</div>
                        <div class="confidence">${{confidenceLevel}} Confidence: ${{confidence}}%</div>
                        <div class="details">
                            Detailed Results:<br>
                            ${{result.all_classes.class_1.name}}: ${{class1Conf}}% | 
                            ${{result.all_classes.class_2.name}}: ${{class2Conf}}%
                        </div>
                    </div>
                `;
            }}

            // Check if camera is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                updateStatus('❌ Camera not supported in this browser', 'error');
                startBtn.disabled = true;
            }}
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process the image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"Original image size: {image.size}, mode: {image.mode}")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        logger.info(f"Raw predictions: {predictions[0]}")
        
        # Get the predicted class and confidence
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])
        
        # For Teachable Machine models, class indices are 0-based
        # Class 0 = First person, Class 1 = Second person
        display_class = predicted_class_index + 1  # Convert to 1-based for display (1 or 2)
        class_name = class_names[predicted_class_index]
        
        # Log the prediction
        logger.info(f"Predicted class index: {predicted_class_index}")
        logger.info(f"Display class: {display_class}")
        logger.info(f"Class name: {class_name}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"All confidences: {[f'{conf:.4f}' for conf in predictions[0]]}")
        
        return jsonify({
            'class': display_class,
            'class_name': class_name,
            'confidence': confidence,
            'raw_predictions': predictions[0].tolist(),
            'all_classes': {
                'class_1': {
                    'name': class_names[0],
                    'confidence': float(predictions[0][0])
                },
                'class_2': {
                    'name': class_names[1], 
                    'confidence': float(predictions[0][1])
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'class_names': class_names
    })

if __name__ == '__main__':
    print("🚀 Starting AI Person Classifier...")
    print("=" * 50)
    
    # Load model on startup
    if load_model_and_setup():
        print("✅ Model loaded successfully!")
        print(f"🎯 Trained to recognize: {class_names[0]} vs {class_names[1]}")
        print("🎥 Camera Classifier is ready!")
        print("📱 Open http://localhost:5000 in your browser")
        print("🔍 Make sure to allow camera access")
    else:
        print("❌ Failed to load model. Please check your keras_model.h5 file.")
        print("💡 Ensure keras_model.h5 is in the same directory as this script.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)