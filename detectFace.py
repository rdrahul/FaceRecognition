#this script should be able to recognize faces from frames
import numpy as np
import cv2


import pickle

labels = {}
with open('pickle/labels.pickle', 'rb') as f: 
    labels = pickle.load( f)
    labels = {v:k for k,v in labels.items() }


#capture the video
cap = cv2.VideoCapture(0)

#using haar classifier
# face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')

#use lbp classifier
face_cascade  = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

sidefaceDetector = cv2.CascadeClassifier('./cascades/data/lbpcascades/lbpcascade_profileface.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect if there is a face in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h , x:x+w]

        id_ , conf =  recognizer.predict( roi_gray )
        if conf >= 65 and conf <=95 : 
        
            font = cv2.FONT_HERSHEY_SIMPLEX
            name  = labels[id_]
            color = ( 255 , 255 , 255 )
            stroke = 2
            cv2.putText( frame , name , (x,y) , font , 1, color , stroke , cv2.LINE_AA )
            path = str('samples/img' +  str(id_) +'.png')
            cv2.imwrite(path,img[y:(y+h), x:(x+w)])



    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
