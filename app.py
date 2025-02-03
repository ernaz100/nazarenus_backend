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
import time
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from pose_network import PoseNetwork
from joints_to_features import prepare_smpl_for_priorMDM
from pose_estimator import PoseEstimator
from diffusion_motion_inbetweening import generate_inbetween_motion, test
from human3dml_util.mini_prior_mdm import mini_prior_mdm

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
# Load environment variables from .env file
load_dotenv()
pose_estimator = PoseEstimator()
joint_mapping = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 22, # left hand
            19: 14, # right collar
            20: 17, # right shoulder
            21: 19, # right elbow
            22: 21, # right wrist
            23: 23, # right hand
        }
joint_mapping_no_hands = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 14, # right collar
            19: 17, # right shoulder
            20: 19, # right elbow
            21: 21, # right wrist
        }
joint_mapping_no_hands_no_pelvis = {
            0: 0,   # left hip 
            1: 3,   # left knee
            2: 6,   # left ankle
            3: 9,  # left foot
            4: 1,   # right hip
            5: 4,   # right knee
            6: 7,   # right ankle
            7: 10,  # right foot
            8: 2,  # spine1
            9: 5,  # spine2
            10: 8,  # spine3
            11: 11, # neck
            12: 14, # head
            13: 12, # left collar
            14: 15, # left shoulder
            15: 17, # left elbow
            16: 19, # left wrist
            17: 13, # right collar
            18: 16, # right shoulder
            19: 18, # right elbow
            20: 20, # right wrist
        }


# Initialize pose network
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
pose_network = PoseNetwork()
checkpoint = torch.load('checkpoints/model_training_batch_256_epochs25.pth', map_location=device, weights_only=True)
pose_network.load_state_dict(checkpoint['model_state_dict'])
pose_network.to(device)
pose_network.eval()


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

@app.route('/estimate_pose', methods=['POST'])
def handle_pose_estimation():
    try:
        data = request.get_json()
        
        if 'joint_positions' not in data:
            return jsonify({'error': 'No joint positions provided'}), 400
            
        joint_positions = np.array(data['joint_positions'])
        selected_joint = data['selected_joint'] 
        
        # Validate input shape
        if joint_positions.shape != (24, 3):
            return jsonify({'error': 'Invalid joint positions format. Expected shape: (24, 3)'}), 400

        # Remap joints to SMPL order
        remapped_joints = remap_joints(joint_positions)
        
        # Transform joints to match SMPL coordinate system
        transformed_joints = transform_input_joints(remapped_joints)
        
        # Prepare input for pose network
        joints_tensor = torch.tensor(transformed_joints, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        joints_tensor = joints_tensor.to(device)
        
        # Get pose parameters from network
        with torch.no_grad():
            pose_params = pose_network(joints_tensor)
            pose_params = pose_params.cpu().numpy().squeeze()  # Remove batch dim

        # Get predicted joints from the pose parameters
        predicted_joints = pose_estimator.forward_kinematics(pose_params)
        # Remap predicted joints to frontend order
        predicted_joints = pose_estimator.remap_joints_to_frontend(predicted_joints)
        
        frontend_pose_params = remap_pose_params_back(pose_params)

        result = {
            'pose_params': frontend_pose_params.tolist(),
            'predicted_joints': predicted_joints.tolist(),
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_from_keyframes', methods=['POST'])
def handle_generate_from_keyframes():
    try:
        data = request.get_json()
        keyframes = data["keyframes"]
        original_keyframes = data.get("originalKeyframes", [])
        number_diffusion_steps: int = int(data["diffusion_steps"])
        motion_editing = True if original_keyframes else False
        # Extract frame indices
        keyframe_indices = [kf["frame"] for kf in keyframes]
        first_keyframe_index = keyframe_indices[0] if keyframe_indices else None
        
        # Determine the range to fill with original keyframes
        start_fill = max(0, first_keyframe_index - 20)
        end_fill = first_keyframe_index + 21  # 20 frames on either side + 1 for the keyframe itself
        
        # Initialize combined data
        max_frame =  196 if original_keyframes else max(kf['frame'] for kf in keyframes) + 1
        combined_data = np.zeros((max_frame, 263))  # Initialize with zeros
        if motion_editing:
            original_motiondata = np.array([kf["motionData"][1:] for kf in original_keyframes])                   
            # Remap each frame of joint positions
            motion_data_remapped = np.array([remap_joints(frame) for frame in original_motiondata])

            keyframe_motion = np.array([kf["motionData"][1:] for kf in keyframes])                   
            keyframe_motion_remapped = np.array([remap_joints(kf) for kf in keyframe_motion])

            motion_data_remapped[first_keyframe_index] = keyframe_motion_remapped[0]    
            # Prepare features for PriorMDM
            features = prepare_smpl_for_priorMDM(motion_data_remapped)
            
            for idx, feature in enumerate(features):
                frame_idx = idx
                if frame_idx > 195:
                    break
                if frame_idx == first_keyframe_index:
                    combined_data[frame_idx] = feature
                if start_fill <= frame_idx < end_fill:
                    continue  # Skip the range around the first keyframe
                combined_data[frame_idx] = feature
        else:
            # Add Keyframe Motion at the correct position
            keyframe_motion = np.array([kf["motionData"][1:] for kf in keyframes])                   
            keyframe_motion_remapped = np.array([remap_joints(kf) for kf in keyframe_motion])
            keyframe_features = prepare_smpl_for_priorMDM(keyframe_motion_remapped, motion_editing = motion_editing)
            for idx, frame in enumerate(keyframes):
                frame_idx = frame['frame']
                combined_data[frame_idx] = keyframe_features[idx]
            

        # Save to .npy file
        output_path = 'static/motion_data.npy'
        np.save(output_path, combined_data)
        
        generated_motion = generate_inbetween_motion(combined_data, keyframe_indices,first_keyframe_index, motion_editing, number_diffusion_steps)[0][0]
        return jsonify({
            'status': 'success',
            'generated_motion': generated_motion.tolist() if generated_motion is not None else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/estimate_sequence', methods = ['POST'])
def handle_sequence_estimation():
    try:
        data = request.get_json()
        keyframes = data["keyframes"]
        original_keyframes = data.get("originalKeyframes", [])
        number_diffusion_steps: int = int(data["diffusion_steps"])

        # let's take the joints from the animation into a ndarray
        sequence_joints = np.array([kf["motionData"][1:] for kf in original_keyframes])      # (sequence_length, 24, 3)             
        for kf in keyframes:
            i = kf["frame"]
            motion = kf["motionData"][1:]
            sequence_joints[i] = np.array(motion)

        # Map the joint order into the smpl format
        smpl_sequence_joints = np.array([remap_joints(kf) for kf in sequence_joints])


        # since mdm uses a different input, we have to convert the joints into their format
        to_human3dml_tensor = prepare_smpl_for_priorMDM(smpl_sequence_joints)

        # We impaint the sequence again using priormdm
        sequence_predicted_joints = mini_prior_mdm(to_human3dml_tensor, number_diffusion_steps)

        return jsonify({
            'status': 'success',
            'generated_motion': sequence_predicted_joints.tolist() if sequence_predicted_joints is not None else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


def remap_joints(input_joints):
        """
        Remap joints from frontend ordering to SMPL ordering
        
        Args:
            input_joints: numpy array of shape (24, 3) in frontend order
        
        Returns:
            remapped_joints: numpy array of shape (24, 3) in SMPL order
        """
        if input_joints.shape != (24, 3):
            raise ValueError(f"Expected input_joints shape (24, 3), got {input_joints.shape}")

        remapped_joints = np.zeros((24, 3))
        for frontend_idx, smpl_idx in joint_mapping.items():
            remapped_joints[smpl_idx] = input_joints[frontend_idx]
        return remapped_joints

def remap_pose_params_back(smpl_pose_params):
    """
    Remap pose parameters from SMPL ordering back to frontend ordering
    
    Args:
        smpl_pose_params: numpy array of shape (72,) in SMPL order
        (first 3 values are global orientation, then 23 joints * 3 rotation params)
    
    Returns:
        frontend_pose_params: numpy array of shape (72,) in frontend order
    """
    if smpl_pose_params.shape != (72,):
        raise ValueError(f"Expected smpl_pose_params shape (72,), got {smpl_pose_params.shape}")

    # Create output array
    frontend_pose_params = np.zeros(72)
    
    for frontend_idx, smpl_idx in joint_mapping.items():
        src_idx = smpl_idx * 3 
        dst_idx = frontend_idx * 3 
        frontend_pose_params[dst_idx:dst_idx+3] = smpl_pose_params[src_idx:src_idx+3]
    
    return frontend_pose_params

def transform_input_joints(joints):
    transformed_joints = joints.copy()
    transformed_joints[:, 1], transformed_joints[:, 2] = - joints[:, 2], joints[:, 1]
    return transformed_joints

if __name__ == "__main__":
    app.run(host=os.getenv('HOST'))

