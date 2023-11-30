from flask import Flask, request, jsonify
import cv2  # You may need this for image preprocessing
import numpy as np

app = Flask(__name__)


def softmax(x):
    z = x - np.max(x, axis = -1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


@app.route('/predict', methods=['POST'])
def predict_number_class():
    # Load from NumPy binary file
    loaded_weights = np.load('./models/number_classify/model_weights.npy')
    loaded_offset = np.load('./models/number_classify/model_offset.npy')

    # Get the image from the request
    image = request.files['image']
    

    # Resize the image to (28, 28) 
    image = cv2.resize(image, (28, 28))
    # Flatten Image to 1x784 vector
    flat_image = image
    
    y_hat = softmax(flat_image @ loaded_weights + loaded_offset)

    # Make a prediction using your ML model
    prediction = np.argmax(y_hat)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})


    
    
if __name__ == '__main__':
    app.run(debug=True)
