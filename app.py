from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("mnist_cnn_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image data from POST request
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # Remove data URL prefix
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Preprocess image
        img = img.convert('L').resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1).astype("float32") / 255
        
        # Make prediction
        pred = model.predict(img_array)
        digit = str(np.argmax(pred))
        return jsonify({'prediction': digit})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)