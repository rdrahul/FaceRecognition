#this script should be able to recognize faces from frames
import numpy as np
import cv2
from lib.recognize import recognize

#capture the video
cap = cv2.VideoCapture(0)

#using haar classifier
# face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')


#using haar classifier
face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')



#use lbp classifier
# face_cascade  = cv2.CascadeClassifier('./cascades/data/lbpcascades/lbpcascade_frontalface_improved.xml')

sidefaceDetector = cv2.CascadeClassifier('./cascades/data/lbpcascades/lbpcascade_frontalface_improved.xml')

img_test = cv2.imread('./test/peter-dinklage.jpg')
#convertTograyscale
gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

# detect if there is a face in the frame
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
i=0
for (x,y,w,h) in faces:
    
    print ("detected")
    img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    roi  = gray [y:y+h, x:x+w]

    #recognize
    id_ , name = recognize ( roi)
    print  ( id_ )
    if  id_ !=-1  :
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = ( 255 , 255 , 255 )
        stroke = 2
        cv2.putText( gray , name , (x,y) , font , 1, color , stroke , cv2.LINE_AA )
        path = str('samples/img' +  str(id_) +'.png')
        cv2.imwrite(path,img)

    i+=1

#detect side faces
# sideFaces = sidefaceDetector.detectMultiScale( gray )
# for (x,y,w,h) in sideFaces:
#     img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    

# Display the resulting frame
cv2.imshow('frame',gray)
cv2.waitKey(3000)

cv2.imwrite('messigray.png',img_test)
cv2.destroyAllWindows()
