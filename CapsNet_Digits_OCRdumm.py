import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import sys
import cv2
import math

K.set_image_data_format('channels_last')


# In[2]:


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def load_model():

	global eval_model	
	model, eval_model, manipulate_model = CapsNet(input_shape=(28,28,1),
		                                          n_class=10,
		                                          routings=3)

	eval_model.load_weights('numericweights.h5')
	

def preprocess(image):
  
  size=(400,400)

  TARGET_PIXEL_AREA = 350000.0

  ratio = float(image.shape[1]) / float(image.shape[0])
  new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
  new_w = int((new_h * ratio) + 0.5)
  image_resized = cv2.resize(image, (new_w,new_h))

  gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
  ret, thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)

  im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

  return(contours, opening)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
 
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
 
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
 
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
 
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# In[23]:


def prediction(contours,opening):
    (cnts, boundingBoxes) = sort_contours(contours)
    threshold_area=400 #Change it accordingly.

    dob=""
    for cnt in cnts:
        area = cv2.contourArea(cnt)         
        if area > threshold_area:    
            x,y,w,h = cv2.boundingRect(cnt)
            img2 = opening[y:y+h, x:x+w]

            size = (28,28)
            
            img3 = cv2.resize(img2, size)
            
            (thresh, im_bw) = cv2.threshold(img3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img4 = im_bw.reshape(1,28,28,1)
            percentage =float((np.sum(img4 == 255))/784.0)            
            if percentage >0.20 and percentage <0.70:
                print (percentage)

                prediction1=""
                prediction1,prediction2= eval_model.predict(img4,
                                    # empty values for the second vector 
                                   batch_size = 32, verbose = True)
                test = np.argmax(prediction1,1)
                prediction_final = test[0]
                dob+=str(prediction_final)+' '
               
    return (dob)


# In[24]:
def make_predictions(img):
    load_model()
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size=(1700,2338)
    thresh = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    #cropped_current_mobilephone = thresh[1180:1225, 910:1300]
    #cropped_noofdependents = thresh[855:895, 1440:1550]
    #cropped_yearsatcurrentresidence = thresh[1110:1150, 1525:1595]
    #cropped_yearsatcurrentcity = thresh[1070:1110, 1525:1595]
    #cropped_current_address_pincode = thresh[1112:1155, 1040:1255]
    #cropped_aadhar = thresh[1140:1190, 1214:1671]
    #cropped_EmployerAddress_currentexperience = thresh[1850:1900, 885:970]
    cropped_digit_dummy = thresh[1250:1300, 1214:1671]
    #cropped_EmployerAddress_Totalexperience = thresh[1850:1900, 1405:1485]

    #cropped_dateofbirth = thresh[855:905, 180:500]
    #mobilecontours, mobileopening = preprocess(cropped_current_mobilephone)
    #dependentscontours, dependentsopening = preprocess(cropped_noofdependents)
    #pincodecontours, pincodeopeing = preprocess(cropped_current_address_pincode)
    #dobcountours, dobopening = preprocess(cropped_dateofbirth)
    #yearscrcountours, yearscropening = preprocess(cropped_yearsatcurrentresidence)
    #aadharcontours,aadharopening = preprocess(cropped_aadhar)
    #totalexperiencecontours,totalexpereienceopening = preprocess(cropped_EmployerAddress_Totalexperience)
    #currentexperiencecontours,currentexpereienceopening = preprocess(cropped_EmployerAddress_currentexperience)
    #currentcityyearscontours, currentcityyearsopening = preprocess(cropped_yearsatcurrentcity)
    dummycontours, dummyopening = preprocess(cropped_digit_dummy)

    #mobile_pred = prediction(mobilecontours, mobileopening )
    #dob_pred = prediction(dobcountours, dobopening )	
    #dependent_pred = prediction(dependentscontours, dependentsopening )
    #pin_pred = prediction(pincodecontours, pincodeopeing )
    #aadhar_pred = prediction(aadharcontours,aadharopening)
    #yearscurrentcity_pred = prediction(yearscrcountours, yearscropening )
    #totalexp = prediction(totalexperiencecontours,totalexpereienceopening)
    #currentexp = prediction(currentexperiencecontours,currentexpereienceopening)
    #currentcityyears_pred = prediction(currentcityyearscontours,currentcityyearsopening)
    dummypred = prediction(dummycontours,dummyopening)
    return dummypred
    #return mobile_pred,dob_pred,dependent_pred,pin_pred,yearscurrentcity_pred,aadhar_pred,totalexp,currentcityyears_pred,currentexp

	#yearsatcurrentcitycontours, yearsatcurrentcityopening =  preprocess(cropped_yearsatcurrentcity)
	#yearsatcurrentresidencecontours, yearsatcurrentresidenceopening =   preprocess(cropped_yearsatcurrentresidence)
	#currentexperiencecontours, currentexperienceopening = preprocess(cropped_EmployerAddress_currentexperience)
	#totalexperiencecontours, totalexpereienceopening  = preprocess(cropped_EmployerAddress_Totalexperience)

img=cv2.imread('ocr ddeddddd-1.jpg')
pred = make_predictions(img)
print(pred)


