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
import ticks as ticks_ocr
import Indian_City_Correction as correction_ocr
import numpy as np
from flask import request
import flask
import requests
import io, os
import difflib
from scipy.misc import *


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)



@app.route("/predict", methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}
    success = False

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "GET":
        
        #read the image in PIL format
        #image = flask.request.files["image"].read()
        #image = imread(io.BytesIO(image))
        url = request.args['url']
        lasturl = url[-3:]
        lasturl = lasturl.lower()
        r = requests.get(url)
        if lasturl == 'pdf':
            open('scannedforocr.pdf', 'wb').write(r.content)
            from wand.image import Image as Img
            with Img(filename='scannedforocr.pdf', resolution=300) as img:
                img.compression_quality = 99
                img.save(filename='imageforocr.jpg')
        else:
            open('imageforocr-0.jpg', 'wb').write(r.content)
        data["dob"] = None
        data["mobile"] = None
        data["no_of_dependent"] = None
        data["pincode"] = None
        data['father_name'] = None
        data['customer_name'] = None
        data['years_at_current_residence'] = None
        data['city'] = None
        data['total_experience'] = None
        data['aadhar'] = None
        data['years_at_current_city'] = None
        data['current_experience'] = None
        data['state'] = None
        #data['dummypred'] = None

        image=cv2.imread('imageforocr-0.jpg')

        mobile_pred,dob_pred,dependent_pred,pin_pred,yearscurrentcity_pred,aadhar,totalexp,yearsatcurrentcity,currentexp=digit_ocr.make_predictions(image)


        # loop over the results and add them to the list of
        # returned predictions
        data['dob']=dob_pred
        data['mobile']=mobile_pred
        data['no_of_dependent']=dependent_pred
        data['pincode']=pin_pred
        data['years_at_current_residence'] = yearscurrentcity_pred
        data['aadhar'] = aadhar
        data['total_experience'] = totalexp
        data['years_at_current_city'] = yearsatcurrentcity
        data['current_experience'] = currentexp
        #data['dummypred'] = dummypred
        persons_name,fathers_name,city,state=characters_ocr.make_predictions(image)
        statename = correction_ocr.statename(state)
        cityname = correction_ocr.correction(city)
        data['father_name']=fathers_name
        data['customer_name']=persons_name
        data['city'] = cityname
        data['state'] = statename

        # indicate that the request was a success
        #data["success"] = True
        success = True
        message = 'ok'
        status = 200
        dicts = ticks_ocr.ticks_prediction(image)
        
        predictdict = {**data,**dicts}
        #print(predictdict)
        try:
            os.remove('imageforocr-0.jpg')
        except OSError:
            pass
        try:
            os.remove('imageforocr-1.jpg')
        except OSError:
            pass
        try:
            os.remove('imageforocr-2.jpg')
        except OSError:
            pass
        try:
            os.remove('scannedforocr.pdf')
        except OSError:
            pass

    # return the data dictionary as a JSON response
    return flask.jsonify(body = predictdict, success = success, message = message, status = status)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    app.run(port=8124)
    
    #print(("* Loading Keras model and Flask starting server..."
    #    "please wait until server has fully started"))
#    load_model()
    

