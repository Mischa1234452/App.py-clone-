from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename


# Load your trained model
model = load_model('/Users/mischa/saved_models/my_model.h5')

app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        result = np.argmax(preds, axis=1)

        # Process your result for human
        if result[0] == 0:
            prediction = 'benign'
        else:
            prediction = 'malignant'
        
        return jsonify(result=prediction)

    return None

if __name__ == '__main__':
    # Choose the port if the default is in use
    port = int(os.environ.get('PORT', 5001))
    app.run(port=port, debug=True)
