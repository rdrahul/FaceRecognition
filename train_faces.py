import os
from PIL import Image
import numpy as np
from lib.detect import face_detect
import pickle
import cv2

#walking the directory
BASE_DIR = os.path.dirname(os.path.abspath( __file__ ) )
imageDir = os.path.join(BASE_DIR , 'train')

#extensions
extns = ['png' , 'jpg' , 'jpeg' , 'webp' , 'PNG' ]


x_train = []
y_train = []


current_id = 0
labelId = {}

for root , dirs , files in os.walk( imageDir):
    for file in files :
        for ext in extns : 
            if ( file.endswith(ext)   ):
                path = os.path.join(root , file)
                label  = os.path.basename( os.path.dirname(path) )


                #check if label exists
                if not label in labelId :
                    labelId[label] = current_id
                    current_id+=1
                id_ = labelId[label]
                
                
                #convert images to grayscale 
                pil_image = Image.open(path).convert("L")
                # size = (700, 700)
                # final_image = pil_image.resize( size , Image.ANTIALIAS)
                
                #convert grayscale image to numpy array
                image_array = np.array(pil_image , "uint8" )
                faces = face_detect( image_array )
                


                for (x,y,w,h) in faces : 
                    #take region of interest
                    roi = image_array[y:y+h , x:x+w]
                    x_train.append(roi)
                    y_train.append(id_)
                


with open('pickle/labels.pickle', 'wb') as labels: 
    pickle.dump( labelId , labels)

print (labelId)
print (y_train)
print ("Training ... ")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train ,  np.array( y_train ))
recognizer.save("trainer.yml")
print ("Training Complete!")
