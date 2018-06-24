import numpy as np
import cv2
import pickle
import os


trainer_path = os.path.join( os.path.abspath( os.path.dirname(os.curdir) ) , 'trainer.yml')
print ( trainer_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(  trainer_path) 


labels = {}
with open('pickle/labels.pickle', 'rb') as f: 
    labels = pickle.load( f)
    labels = {v:k for k,v in labels.items() }


def recognize(image) :


    id_ , conf =  recognizer.predict( image )
    if conf >= 65 and conf <=95 : 

        return  ( id_ , labels[id_] )
    return (-1 , None)
    