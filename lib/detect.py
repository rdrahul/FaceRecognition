import cv2 

#using haar classifier
face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')



def face_detect( image ) : 

    faces  = face_cascade.detectMultiScale( image  , 1.3 , 3)

    return faces


