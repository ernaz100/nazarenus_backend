import json

from flask import Flask, request, jsonify, send_file, make_response, url_for, stream_with_context, Response
from flask_mail import Mail, Message
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
import torch
import anthropic
from models.number_classify.MLP import MLP
from torchvision import transforms
from models.number_classify.SimpleCNN import SimpleCNN
from dotenv import load_dotenv
import os

from radiance import generate_clip

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
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


@app.route('/generate-message', methods=['POST'])
@cross_origin()
def generate_message():
    try:
        # Ensure that the request contains necessary data
        if 'content' not in request.json:
            return jsonify({'error': 'Content field is required'}), 400

        # API Key for Anthropictext API
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'API Key not found'}), 500
        client = anthropic.Anthropic(api_key=api_key, )

        message = client.beta.tools.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system="You are my writing assistant. I want you to generate a project outline. Make sure to take the tone (formal, scientific, comedic etc.) into account.",
            tools=[
                {
                    "name": "get_outline",
                    "description": "Get the outline for a writing project",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The proposed title for the writing project",
                            },
                            "outline": {
                                "type": "string",
                                "description": "The proposed outline for the writing project",
                            },

                        },
                        "required": ["title", "outline"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": request.json['content']}
            ]
        )
        if message:
            if (type(message.content[0]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[0].input}), 200
            elif (type(message.content[1]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[1].input}), 200
            elif (type(message.content[2]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[2].input}), 200

        else:
            return jsonify({'error': 'Failed to generate message in backend'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/complete-sentence', methods=['POST'])
@cross_origin()
def complete_sentence():
    try:
        # Ensure that the request contains necessary data
        if 'content' not in request.json:
            return jsonify({'error': 'Content field is required'}), 400

        # API Key for Anthropictext API
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'API Key not found'}), 500
        client = anthropic.Anthropic(api_key=api_key, )

        message = client.beta.tools.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            tools=[
                {
                    "name": "complete_sentence",
                    "description": "Complete the last sentence of the text that I'm writing.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "end_of_sentence": {
                                "type": "string",
                                "description": "The end of the sentence. Just the part that we append to the existing text.",
                            },

                        },
                        "required": ["end_of_sentence"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": request.json['content']}
            ]
        )
        if message:
            if (type(message.content[0]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[0].input}), 200
            elif (type(message.content[1]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[1].input}), 200
            elif (type(message.content[2]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[2].input}), 200

        else:
            return jsonify({'error': 'Failed to generate message in backend'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rephrase-sentence', methods=['POST'])
@cross_origin()
def rephrase_sentence():
    try:
        # Ensure that the request contains necessary data
        if 'content' not in request.json:
            return jsonify({'error': 'Content field is required'}), 400

        # API Key for Anthropictext API
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'API Key not found'}), 500
        client = anthropic.Anthropic(api_key=api_key, )

        message = client.beta.tools.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system="You are my writing assistant. I want you to rephrase the text that I'm writing. I want you to give me 3 different versions. Make sure to take the tone (formal, scientific, comedic etc.) into account.",

            tools=[
                {
                    "name": "rephrase_sentence",
                    "description": "Please rephrase the text that I'm writing. I want you to give me 3 different versions",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "rephrased_v1": {
                                "type": "string",
                                "description": "The first rephrased version of the provided text.",
                            },
                            "rephrased_v2": {
                                "type": "string",
                                "description": "The second rephrased version of the provided text.",
                            },
                            "rephrased_v3": {
                                "type": "string",
                                "description": "The third rephrased version of the provided text.",
                            },

                        },
                        "required": ["rephrased_v1, rephrased_v2, rephrased_v3"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": request.json['content']}
            ]
        )
        if message:
            if (type(message.content[0]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[0].input}), 200
            elif (type(message.content[1]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[1].input}), 200
            elif (type(message.content[2]) == anthropic.types.beta.tools.tool_use_block.ToolUseBlock):
                return jsonify({'message': message.content[2].input}), 200

        else:
            return jsonify({'error': 'Failed to generate message in backend'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/getVideos', methods=["POST", "OPTIONS"])
@cross_origin()
def send_videos():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_preflight_response()
    elif request.method == "POST":
        data = request.get_json()
        youtube_link = data.get('link', '')
        count = data.get('count', 4)  # Get the number of videos to generate

        def generate():
            for video_path, title, description in generate_clip(youtube_link):
                if not os.path.exists(video_path):
                    yield json.dumps({"error": f"Video file {title} not found"}) + '\n'
                    continue

                # Extract the first frame as thumbnail
                video = cv2.VideoCapture(video_path)
                success, frame = video.read()
                if success:
                    thumbnail_path = video_path.replace('.mp4', '_thumbnail.jpg')
                    cv2.imwrite(thumbnail_path, frame)
                video.release()

                # Extract the directory and filename from video_path and thumbnail_path
                video_dir = os.path.dirname(video_path)
                video_filename = os.path.basename(video_path)
                thumbnail_filename = os.path.basename(thumbnail_path)

                # Generate URLs with directory and filename
                video_url = url_for('serve_video', directory=video_dir, filename=video_filename, _external=True)
                thumbnail_url = url_for('serve_thumbnail', directory=video_dir, filename=thumbnail_filename, _external=True)

                video_data = {
                    'id': title,  # You might want to generate a unique ID
                    'videoUrl': video_url,
                    'thumbnailUrl': thumbnail_url,
                    'title': title,
                    'description': description,
                }
                yield json.dumps(video_data) + '\n'

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))


@app.route('/video/<path:directory>/<filename>')
def serve_video(directory, filename):
    # Construct the full path using the directory and filename
    file_path = os.path.join("/" + directory, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='video/mp4')
    else:
        return {"error": "File not found"}, 404

@app.route('/thumbnail/<path:directory>/<filename>')
def serve_thumbnail(directory, filename):
    # Construct the full path using the directory and filename
    file_path = os.path.join("/" + directory, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        app.logger.warning(f"Filepath: {file_path}")
        return {"error": "File not found"}, 404

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host=os.getenv('HOST'))

import time
def generate():
    for index in range(1, 3):
        time.sleep(3)  # Simulate processing time
        video_directory = 'clips/The German Problem'
        video_filename = f'highlight_{index}_The German Problem.mp4'
        video_path = os.path.join(video_directory, video_filename)

        if not os.path.exists(video_path):
            yield json.dumps({"error": f"Video file {video_filename} not found"}) + '\n'
            continue

        # Extract the first frame as thumbnail
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        if success:
            thumbnail_path = video_path.replace('.mp4', '_thumbnail.jpg')
            cv2.imwrite(thumbnail_path, frame)
        video.release()

        # Get the URLs for the video and thumbnail
        video_url = url_for('serve_video', filename=os.path.basename(video_path), _external=True)
        thumbnail_url = url_for('serve_thumbnail', filename=os.path.basename(thumbnail_path), _external=True)

        video_data = {
            'id': str(index),
            'videoUrl': video_url,
            'thumbnailUrl': thumbnail_url,
            'title': f'Generated Video {index}',
            'description': f'Description of generated video {index}'
        }

        yield json.dumps(video_data) + '\n'
