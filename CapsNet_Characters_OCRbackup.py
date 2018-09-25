
# coding: utf-8

# In[1]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="specify test image",default="")
parser.add_argument("--epochs", help="specify number of epochs",default=10)
parser.add_argument("--test_size", help="specify test size",default=.01)
args = parser.parse_args()


import os
#os.system('git clone https://github.com/XifengGuo/CapsNet-Keras')
#os.chdir('CapsNet-Keras')


# In[4]:


import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')
seed=42

# In[5]:


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


# In[6]:

def train(model, data):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger('./log.csv')
    #tb = callbacks.TensorBoard('/tensorboard-logs',batch_size=100, histogram_freq=int('store_true'))
    checkpoint = callbacks.ModelCheckpoint('./weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.1 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, 100, 0.1),
                        steps_per_epoch=int(y_train.shape[0] / 100),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights('./trained_model.h5')
    #print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model




def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) +         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


# In[6]:


def load_data():
    # the data, shuffled and split between train and test sets
    from sklearn.model_selection import train_test_split
    
    dataset = np.loadtxt('A_Z Handwritten Data.csv', delimiter=',')
    X = dataset[:,0:784]
    Y = dataset[:,0]
    
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=args.test_size, random_state=seed)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


# In[9]:


# load data
#(x_train, y_train), (x_test, y_test) = load_data()

    # define model
model, eval_model, manipulate_model = CapsNet(input_shape=(28,28,1),
                                                  n_class=26,
                                                  routings=3)
#model.summary()


from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

#train(model=model, data=((x_train, y_train), (x_test, y_test)))
model.load_weights('alphabetweights.h5')
# In[14]:


import cv2
sample_path=args.image
img=cv2.imread(sample_path)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
size=(1700,2338)
thresh = cv2.resize(img, size, interpolation = cv2.INTER_AREA)


# In[15]:


cropped_Applicant_Name = thresh[765:815, 180:1685]
cropped_fathersname = thresh[810:862, 180:1480]
cropped_applicationdate = thresh[535:585, 1450:1685]
cropped_pan = thresh[725:775, 1235:1685]

cropped_dateofbirth = thresh[855:905, 180:500]
cropped_noofdependents = thresh[855:895, 1440:1550]
cropped_current_address_line1 = thresh[950:1000, 180:1480]
cropped_current_address_line2 = thresh[990:1035, 180:1480]
cropped_current_address_line3 = thresh[1028:1072, 180:1480]
cropped_current_address_landmark = thresh[1070:1115, 180:1300]
cropped_current_address_city = thresh[1112:1155, 180:930]
cropped_current_address_state = thresh[1145:1190, 180:705]
cropped_current_address_country = thresh[1140:1190, 770:1075]
cropped_current_address_pincode = thresh[1112:1155, 1040:1255]
cropped_aadhar = thresh[1140:1190, 1214:1671]
cropped_current_telephone = thresh[1180:1225, 180:770]
cropped_current_mobilephone = thresh[1180:1225, 910:1300]
cropped_emailid = thresh[1220:1270, 180:1671]
cropped_yearsatcurrentcity = thresh[1070:1115, 1580:1670]
cropped_yearsatcurrentresidence = thresh[1110:1155, 1585:1670]
cropped_current_address_line1 = thresh[950:1000, 180:1480]
cropped_current_address_line2 = thresh[990:1035, 180:1480]
cropped_current_address_line3 = thresh[1028:1072, 180:1480]
cropped_current_address_landmark = thresh[1070:1115, 180:1300]
cropped_current_address_city = thresh[1112:1155, 180:930]
cropped_current_address_state = thresh[1145:1190, 180:705]
cropped_current_address_country = thresh[1140:1190, 770:1075]
cropped_current_address_pincode = thresh[1112:1155, 1040:1255]
cropped_aadhar = thresh[1140:1190, 1214:1671]
cropped_current_telephone = thresh[1180:1225, 180:770]
cropped_current_mobilephone = thresh[1180:1225, 910:1300]
cropped_emailid = thresh[1220:1270, 180:1671]
cropped_yearsatcurrentresidence = thresh[1110:1150, 1525:1595]
cropped_yearsatcurrentcity = thresh[1070:1110, 1525:1595]
#cropped_yearsatcurrentcity = thresh[1070:1115, 1580:1670]
#cropped_yearsatcurrentresidence = thresh[1110:1155, 1585:1670]
cropped_EmployerName = thresh[1900:1955, 180:1480]
cropped_EmployerAddress_line1 = thresh[1940:1990, 180:1480]
cropped_EmployerAddress_line2 = thresh[1985:2030, 180:1480]
cropped_EmployerAddress_line3 = thresh[2020:2065, 180:1480]
cropped_EmployerAddress_landmark = thresh[2050:2100, 180:1480]
cropped_EmployerAddress_city = thresh[2095:2140, 180:950]
cropped_EmployerAddress_pincode = thresh[2095:2140, 1000:1480]
cropped_EmployerAddress_state = thresh[2132:2180, 180:950]
cropped_EmployerAddress_country = thresh[2132:2180, 1000:1670]
cropped_EmployerAddress_telephone = thresh[2175:2220, 180:1000]
cropped_EmployerAddress_official_emailID = thresh[2220:2260, 180:1671]
cropped_EmployerAddress_currentexperience = thresh[1850:1900, 885:970]
cropped_EmployerAddress_Totalexperience = thresh[1850:1900, 1405:1485]
cropped_test_alphabet = thresh

from PIL import Image
import sys
import cv2
import numpy as np
import math


# In[20]:


def preprocess(image):
  
  size=(400,400)

  TARGET_PIXEL_AREA = 350000.0

  ratio = float(image.shape[1]) / float(image.shape[0])
  new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
  new_w = int((new_h * ratio) + 0.5)
  image_resized = cv2.resize(image, (new_w,new_h))
#plt.imshow(cropped_dateofbirth_resized)
# noise removal
  gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
  ret, thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#plt.imshow(thresh1)
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)

  im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  #img_contour=cv2.drawContours(image_resized.copy(), contours, -1, (0,255,0), 3)
  return(contours, opening)


# In[21]:


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


# In[22]:


letter_count = dict([(i-65,chr(i)) for i in range(65,91)])
#letter_count = dict([(i,chr(i)) for i in range(65,91)])

# In[24]:


def prediction(contours,opening):
    (cnts, boundingBoxes) = sort_contours(contours)
    threshold_area=400 #Change it accordingly.

    dob=""
    for cnt in cnts:
        area = cv2.contourArea(cnt)         
        if area > threshold_area:    
            x,y,w,h = cv2.boundingRect(cnt)
            img2 = opening[y:y+h, x:x+w]

            #img3 = Image.fromarray(img2)
            #img3 = cv2.bilateralFilter(img2, 11, 17, 17)
            #edges = cv2.Canny(img3,100,300,apertureSize = 3)
            size = (28,28)
            #img3 = imresize(img2, size)
            img3 = cv2.resize(img2, size)
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            #threshyou = cv2.dilate(img3,kernel,iterations = 3)
            #threshblur = cv2.bilateralFilter(threshyou, 11, 17, 17)
            # = cv2.convexHull(cnt)
            (thresh, im_bw) = cv2.threshold(img3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img4 = im_bw.reshape(1,28,28,1)
            percentage =float((np.sum(img4 == 255))/784.0)            
            if percentage >0.20 and percentage <0.70:
                print (percentage)
                #plt.imshow(img3,cmap='gray')
                #plt.show()
                prediction1=""
                #prediction1=  model.predict(img3.reshape(1,28,28,1))#pytesseract.image_to_string(img2,   config="-c tessedit_char_whitelist=0123456789")

                #prediction1 = model.predict([img3.reshape(1,28,28,1), np.zeros((data_test.shape[0],10))]
                prediction1,prediction2= eval_model.predict(img4,
                                    # empty values for the second vector 
                                   batch_size = 32, verbose = True)
                test = np.argmax(prediction1,1)
                prediction_final = test[0]
                #if prediction_final < 10:
                #	dob+=str(prediction_final)+' '
               	#else:
               	#	prediction_final = 55+prediction_final
               	#	dob+=str(letter_count[prediction_final])+' '

                dob+=str(letter_count[prediction_final])+' '




                #print (prediction1)
                #plt.imshow(img3)
                #plt.show()
    return (dob)


# In[25]:


applicantnamecountours, applicantnameopening = preprocess(cropped_Applicant_Name)
fathersnamecountours, fathersnameopening = preprocess(cropped_fathersname)
employercitycountours,employercityopening = preprocess(cropped_EmployerAddress_city)
employernamecountours,employernameopening = preprocess(cropped_EmployerName)
mobilecountours,mobileopening = preprocess(cropped_current_mobilephone)
cropped_test_alphabetcountours,cropped_test_alphabetopening = preprocess(cropped_test_alphabet)


# In[28]:



#fathersnamepredicted = prediction(fathersnamecountours, fathersnameopening)
#print("Father's name is ",fathersnamepredicted)
#applicantnamepredicted = prediction(applicantnamecountours, applicantnameopening)
#print("Applicant name is ",applicantnamepredicted)
#employercitypredicted = prediction(employercitycountours, employercityopening)

#print("employer city name is ",employercitypredicted)
employernamepredicted = prediction(cropped_test_alphabetcountours, cropped_test_alphabetopening)
print("employer name is ",employernamepredicted)

#pan = prediction(mobilecountours, mobileopening)
#print("pan name is",pan)


