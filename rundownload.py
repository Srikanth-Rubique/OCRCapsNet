# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
# import the necessary packages
#from keras.applications import ResNet50
import cv2
import CapsNet_Digits_OCR as digit_ocr
import CapsNet_Characters_OCR as characters_ocr
import numpy as np
import flask
from flask import request
import requests
import io,os
from scipy.misc import *


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)



@app.route("/predict", methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "GET":
        
            # read the image in PIL format
            #image = flask.request.files["image"].read()
            #image = imread(io.BytesIO(image))
        url = request.args['url']
        r = requests.get(url)
        #print(r.content)

        open('imageforocr.jpg', 'wb').write(r.content)
        print("file saved")
        print("deleting file")
        os.remove('imageforocr.jpg')

        

    # return the data dictionary as a JSON response
    return "Hello"

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
#    load_model()
    app.run()

