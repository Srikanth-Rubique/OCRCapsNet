import cv2
import numpy as np
#sample_path='application_form-5a096c7c-fb0-1.jpg'

#img=cv2.imread(sample_path)44

def ticks_prediction(img):
  imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  size=(1700,2338)
  thresh = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

  #type of loan
  personal_loan = thresh[570:600, 40:75]
  business_loan = thresh[570:600, 170:215]
  large_ticket_business_loan = thresh[570:600, 300:340]
  self_employed_loan = thresh[575:600, 540:570]

  #plt.imshow(cropped_self_employed_loan)

  #title
  mr = thresh[730:770, 180:224]
  ms = thresh[730:770, 250:297]
  mrs = thresh[730:770, 330:370]
  #plt.imshow(cropped_title_ms)

  #gender
  male = thresh[850:900, 580:630]
  female = thresh[848:900, 625:670]
  #plt.imshow(cropped_female)

  #marital status
  single = thresh[855:900, 995:1050]
  maried = thresh[855:900, 1110:1153]
  #plt.imshow(cropped_single)

  #education
  undergraduate = thresh[898:940, 177:216]
  graduate = thresh[898:940, 325:370]
  postgraduate = thresh[898:940, 440:480]
  others = thresh[898:940, 650:694]
  #plt.imshow(cropped_education_postgrad)

  #address
  owned= thresh[1261:1295, 216:260]
  parental= thresh[1261:1295, 360:405]
  company_provided= thresh[1261:1295, 510:560]
  rental= thresh[1261:1295, 730:780]
  #plt.imshow(cropped_address_rental)

  #occupation
  salaried = thresh[1600:1630, 360:405]
  self_employed = thresh[1600:1630, 510:555]
  self_employed_professional = thresh[1600:1630, 700:740]
  retired = thresh[1600:1630, 990:1035]
  #housewife = thresh[1600:1630, 1210:1260]
  #student = thresh[1605:1630, 1360:1405]
  other = thresh[1605:1630, 1470:1520]
  #plt.imshow(cropped_occuaption_salaried)


  from collections import OrderedDict
  tick_mark_loan_dictionary=OrderedDict([('Personal loan',personal_loan),
                        ('Business loan',business_loan),
                        ('Large ticket business loan',large_ticket_business_loan),
                        ('Self employed loan',self_employed_loan)])

  tick_mark_title_dictionary=OrderedDict([('Mr',mr),
                        ('Ms',ms),
                        ('Mrs',mrs)])

  tick_mark_gender_dictionary=OrderedDict([('Male',male),
                        ('Female',female)])

  tick_mark_marital_dictionary=OrderedDict([('Single',single),
                        ('Maried',maried)])

  tick_mark_education_dictionary=OrderedDict([('Undergraduate',undergraduate),
                        ('Graduate',graduate),
                        ('Postgraduate',postgraduate),
                        ('Others',others)])

  tick_mark_address_dictionary=OrderedDict([('Owned',owned),
                        ('Parental',parental),
                        ('Company provided',company_provided),
                        ('Rental',rental)])

  tick_mark_occupation_dictionary=OrderedDict([('Salaried',salaried),
                        ('Self employed',self_employed),
                        ('Self employed professional',self_employed_professional),
                        ('Retired',retired),
                        ('Other',other)])



  tick_mark_dictionary=OrderedDict([('loantype',tick_mark_loan_dictionary),
                                   ('title',tick_mark_title_dictionary),
                                   #('gender',tick_mark_gender_dictionary),
                                   ('marital_status',tick_mark_marital_dictionary),
                                   ('educationtype',tick_mark_education_dictionary),
                                   ('addresstype',tick_mark_address_dictionary),
                                   ('occupation',tick_mark_occupation_dictionary)])
  import math
  def calc_tick_area(image):
      #image=cropped_student
      size=(400,400)

      TARGET_PIXEL_AREA = 350000.0

      ratio = float(image.shape[1]) / float(image.shape[0])
      new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
      new_w = int((new_h * ratio) + 0.5)
      image_resized = cv2.resize(image, (new_w,new_h))
      #plt.imshow(cropped_dateofbirth_resized)
      # noise removal
      gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
      ret, thresh1 = cv2.threshold(gray,500,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      kernel = np.ones((3,3),np.uint8)
      opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)
      #n_white_pix = np.sum(opening == 255)
      #n_black_pix = np.sum(opening == 0)
      #print float(n_black_pix/n_white_pix)
      im2, cnts, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      max=0
      for cnt in cnts:
              area = cv2.contourArea(cnt)
              if area>max:
                  max=area
      return max
  def calc_male_female (tick_mark_gender_dictionary):
      first=True
      area=0
      for key in tick_mark_gender_dictionary.keys():
          image=tick_mark_gender_dictionary[key]
          size=(400,400)

          TARGET_PIXEL_AREA = 350000.0

          ratio = float(image.shape[1]) / float(image.shape[0])
          new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
          new_w = int((new_h * ratio) + 0.5)
          image_resized = cv2.resize(image, (new_w,new_h))
          #plt.imshow(cropped_dateofbirth_resized)
          # noise removal
          gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
          ret, thresh1 = cv2.threshold(gray,500,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
          kernel = np.ones((3,3),np.uint8)
          opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)
          n_white_pix = np.sum(opening == 255)
          n_black_pix = np.sum(opening == 0)
          percentage=float(n_black_pix/n_white_pix)
          if percentage<area:
              area=percentage
              first= not first
          return first

  min=1000000
  area=None
  parent_key=None
  child_key=None
  dicts = {}
  for key in tick_mark_dictionary.keys():  
      min=1000000
      for sub_key in tick_mark_dictionary[key].keys():
         # print sub_key
          #print tick_mark_dictionary[key][i]
          area=calc_tick_area(tick_mark_dictionary[key][sub_key])
          if area < min:
              min=area
              parent_key=key
              child_key=sub_key
      #print (min)
      dicts[parent_key] = child_key
      #print (parent_key)
      #print (child_key)
      #print ('************************************')
  dicts['gender'] = 'Female'
  if calc_male_female(tick_mark_gender_dictionary):
    dicts['gender'] = 'Male'
    #print ('male')
  return dicts

#('gender',tick_mark_gender_dictionary)
#print(dicts)


