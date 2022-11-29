# coding=utf-8
#PASO 1: Importar todas las dependencias del proyecto
from flask import Flask, request, jsonify
import efficientnet.tfkeras as efn
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from flask_cors import CORS


AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__)


#PASO 2: Cargar el modelo con los pesos pre entrenado
with open('plant_model.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# cargar los pesos en el modelo
model.load_weights("plant_model.h5")

#model = efn.EfficientNetB7(weights="plant_model.h5", include_top=False)

#Declaracion de Funciones y carga de datos
def decode_image(filename, label=None, image_size=(256, 256)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    

def format_path(st):
    GCS_DS_PATH = '/var/www/html/images/' + st + '.jpg'
    return GCS_DS_PATH 





def predictModel():

    directoryUpload = np.array(os.listdir('/var/www/html/images/'))
    list_images = np.array(['/var/www/html/images/' + sub for sub in directoryUpload])
    
    numFiles = 0

    for path in os.listdir('/var/www/html/images/'):
        if os.path.isfile(os.path.join('/var/www/html/images/', path)):
            numFiles += 1


    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(list_images,)
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(1)
    )
    

    # initialize list elements
    array1 = np.array(np.core.char.rstrip(directoryUpload,'.jpg'))
    array2 = np.zeros((numFiles, 4))
    array2.fill(0.25)


    arrayTest = np.column_stack([array1, array2])
    
    template = pd.DataFrame(arrayTest, columns=['image_id',  'healthy', 'multiple_diseases', 'rust', 'scab'])
    print(template)
    
    probs = model.predict(test_dataset)

    template.loc[:, 'healthy':] = probs
    return template


#PASO 3: Crear la API con Flask

#Crear una aplicación de Flask
app = Flask(__name__)
CORS(app)


@app.route("/api/v1/", methods=["POST"])
def predictAll():
    #Hacer la predicción utilizando el modelo pre entrenado
    
    sub = predictModel()
    prediction = sub.loc[sub['image_id']]
    #Devolver la predicción al usuario
    jsonrespons = sub.to_json(orient = 'index')
    print(sub)
    print(type(sub))
    idMax = int(np.argmax(float(sub)))
    z = {'PredictCategory' : sub.index[idMax]}
    jsonrespons.update(z)
    print(jsonrespons)
    return jsonrespons


#Definir la función de clasificación de imágenes
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
 
    
    #Hacer la predicción utilizando el modelo pre entrenado
    sub = predictModel()
    #Devolver la predicción al usuario
    print(type(img_name))
    ind = sub.loc[sub['image_id'] == re.sub('\.jpg', '', img_name)].index
    re.sub('\.jpg', '', img_name)
    print(sub)
    prediction = sub.loc[sub['image_id'] == re.sub('\.jpg', '', img_name)].loc[ind[0]].drop(['image_id'])


    #print("La prediccion es: " + str(prediction))
    #print(type(prediction))
    idMax = int(np.argmax(prediction))

    
    
    jsonpredictionponse = {
        'nameImage' : img_name,
        'PredictCategory' : prediction.index[idMax],
        'Probabilty' : str(prediction[idMax]),
        'probs' : [
                {'healthy' : prediction['healthy']},
                {'multiple_diseases' : prediction['multiple_diseases']},
                {'rust' : prediction['rust']},
                {'scab' : prediction['scab']}
            ]
    }
    
    return jsonpredictionponse

#Iniciar la aplicación de Flask
app.run(ssl_context=('cert.pem','key.pem'),port=5000, debug=False)

