from flask import Blueprint, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import pydicom as dicom
import numpy as np
from PIL import Image
import cv2
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


main = Blueprint('main', __name__)
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
array = None

def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array) 
    img2 = img_array.astype(float) 
    img2 = (np.maximum(img2,0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show 

def preprocess(array):
     array = cv2.resize(array , (512 , 512))
     array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
     array = clahe.apply(array)
     array = array/255
     array = np.expand_dims(array,axis=-1)
     array = np.expand_dims(array,axis=0)
     return array

def load_model():
    model_cnn = tf.keras.models.load_model('app/WilhemNet_86.h5')
    return model_cnn

def grad_cam(array): 
    img = preprocess(array)
    model = load_model()
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:,argmax]
    last_conv_layer = model.get_layer('conv10_thisone')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    for filters in range(64):
        conv_layer_output_value[:,:,filters] *= pooled_grads_value[filters]
    #creating the heatmap
    heatmap = np.mean(conv_layer_output_value, axis = -1)
    heatmap = np.maximum(heatmap, 0)# ReLU
    heatmap /= np.max(heatmap)# normalize
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array , (512 , 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency,img2)  
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:,:,::-1]

def predict(array): 
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
    model = load_model()
    #model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img))*100
    label = ''
    if prediction == 0:
        label = 'bacteriana'
    if prediction == 1:
        label = 'normal'
    if prediction == 2:
        label = 'viral'
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap    
    heatmap = grad_cam(array)
    return(label, proba, heatmap)

def run_model():
    global array
    label, poba, heatmap = predict(array) 
    print('OK')
    return label, poba, heatmap

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict_route():
    global array
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        if filepath:            
            array, img2show = read_jpg_file(filepath)
            print('Filepath')
        # Aquí irá la lógica para predecir
        result, probability, heatmap  = run_model()
        cv2.imwrite(f'app/static/uploads/heat_{filename}',heatmap)
        # Aquí generamos la URL para la imagen de heatmap (usando la misma imagen para simplificar)
        heatmap_url = url_for('static', filename=f'uploads/heat_{filename}')
        return jsonify({'result': result, 'probability': probability, 'heatmap_url': heatmap_url})
    
    return jsonify({'success': False, 'message': 'File not allowed'})

