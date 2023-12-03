from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = False

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator

def preprocess_image(image):
    try:
        # Resize the image to 28 × 28 pixels
        image = cv2.resize(image, (28, 28))

        # Invert pixel values
        image = 255 - image
        # Convert pixel values to floating-point numbers
        image = image.astype(np.float32)
        # Normalize pixel values to a range between 0 and 1
        image /= 255.0
        return np.array([image])
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict_number_class():
    app.logger.warning("Prediction:")
    try:
        # Get the image file from the request
        image_file = request.files['image']
        #image_file.save("./uploads/" + image_file.filename)
        # Read the image file
        image_data = image_file.read()
        # Decode the image data and convert to NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            #cv2.imwrite('./uploads/preprocessed_image.png', preprocessed_image[0] * 255)
            if preprocessed_image is not None:
                # Flatten Image to 1x784 vector
                flat_image = np.reshape(preprocessed_image, (preprocessed_image.shape[0], -1)).astype('float')

                # Load model weights and offset
                loaded_weights = np.load('./models/number_classify/model_weights.npy')
                loaded_offset = np.load('./models/number_classify/model_offset.npy')
                y_hat = softmax(flat_image @ loaded_weights + loaded_offset)
                print(y_hat)
                # Make a prediction using your ML model
                predicted_class = np.argmax(y_hat)
                #Return y_hat
                return jsonify(y_hat.tolist())
                # Return only the predicted number as an integer
                #return str(predicted_class)
            else:
                return "Error in preprocessing the image", 500
        else:
            return "Invalid image format", 400

    except Exception as e:
        app.logger.warning(f"Error in predict_number_class: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=False)
