from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
import torch
from models.number_classify.MLP import MLP
from torchvision import transforms
from models.number_classify.SimpleCNN import SimpleCNN

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
def preprocess_image(image):
    try:
        # Define the transformation for the input image
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((28, 28)),  # Resize to 28x28 pixels
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
        ])
        # Invert pixel values
        image = 255 - image

        image_tensor = transform(image)  # Apply transformation
        return image_tensor.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_number_class():
    # Load MLP model
    mlp_model_path = 'models/number_classify/mlp.pth'
    mlp_model = MLP()
    checkpoint_mlp = torch.load(mlp_model_path)
    mlp_model.load_state_dict(checkpoint_mlp['model_state_dict'])
    mlp_model.eval()

    # Load CNN model
    cnn_model_path = 'models/number_classify/cnn.pth'
    cnn_model = SimpleCNN()
    checkpoint_cnn = torch.load(cnn_model_path)
    cnn_model.load_state_dict(checkpoint_cnn['model_state_dict'])
    cnn_model.eval()

    try:
        # Get the image file from the request
        image_file = request.files['image']
        # Read the image file
        image_data = image_file.read()
        # Decode the image data and convert to NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Convert image array to grayscale
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # Preprocess the image for both models
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                # Perform inference with MLP model
                with torch.no_grad():
                    mlp_predictions = torch.softmax(mlp_model(preprocessed_image), dim=1).tolist()

                # Perform inference with CNN model
                with torch.no_grad():
                    cnn_predictions = torch.softmax(cnn_model(preprocessed_image), dim=1).tolist()

                # Combine predictions into a single JSON response
                response = {
                    'mlp_predictions': mlp_predictions,
                    'cnn_predictions': cnn_predictions
                }
                return jsonify(response)
            else:
                return "Error in preprocessing the image", 500
        else:
            return "Invalid image format", 400

    except Exception as e:
        app.logger.warning(f"Error in predict_number_class: {e}")
        return "Internal Server Error", 500

@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

# Configure Flask-Mail settings using environment variables
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS').lower() in ['true', '1', 'yes']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)
@app.route('/contact', methods=['POST'])
@cross_origin()
def contact_form():
    name = request.form.get('name')
    concern = request.form.get('concern')
    message = request.form.get('message')
    contact = request.form.get('contact')

    # Send email
    try:
        subject = f"Contact Form: Message from {name} - {concern}"
        body = f"Contact Info: {contact} Message: {message}"

        msg = Message(subject=subject,
                      sender='nazarenusen@gmail.com',
                      recipients=['nazarenuseric@gmail.com'])
        msg.body = body

        mail.send(msg)

        return jsonify({'message': 'Email sent successfully!'}), 200

    except Exception as e:
        app.logger.warning(f"Error in contact: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host=os.getenv('HOST'))
