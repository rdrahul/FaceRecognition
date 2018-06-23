#this script should be able to recognize faces from frames
import numpy as np
import cv2

#capture the video
cap = cv2.VideoCapture(0)

#using haar classifier
# face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')

#use lbp classifier
face_cascade  = cv2.CascadeClassifier('./cascades/data/lbpcascades/lbpcascade_frontalface_improved.xml')

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

    #detect side faces
    # sideFaces = sidefaceDetector.detectMultiScale( gray )
    # for (x,y,w,h) in sideFaces:
    #     img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
