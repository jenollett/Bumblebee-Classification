from flask import Flask, request, jsonify
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
CLASS_NAMES = ["Bombus bohemicus", "Bombus campestris", "Bombus hortorum", "Bombus humilis", "Bombus hypnorum", "Bombus jonellus", "Bombus lapidarius", "Bombus lucorum", "Bombus monticola", "Bombus muscorum", "Bombus pascuorum", "Bombus pratorum", "Bombus ruderatus", "Bombus rupestris", "Bombus soroeensis", "Bombus sylvarum", "Bombus sylvestris", "Bombus terrestris", "Bombus vestalis"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('D:/Jen/Documents/Dissertation/Serve_Models/UK_weights3')

@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if len(request.files) < 1:
        return jsonify({'error':'No posted images.'})
    files = request.files
    confidences = [0 for i in range(19)]
    for file in files.values():
        if file.filename == '':
            return jsonify({'error':'Empty filename submitted.'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("Processing:"+filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            x = []
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            img = Image.open(BytesIO(file.read()))
            img.load()
            img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x)
            
            for i in range(len(pred)):
                confidences = np.add(pred[i], confidences)
        else:
            return jsonify({'error':'File has invalid extension'})
        
    indexes = np.argsort(confidences)[::-1]
    result = [[CLASS_NAMES[i], (confidences[i]/len(request.files))] for i in indexes]
        
    response = {'pred':result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)