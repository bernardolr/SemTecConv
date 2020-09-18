import cv2
import os
import numpy as np

labels = []
facesData = []
label = 0

dataPath = 'data'
namesList = os.listdir(dataPath)

for i in namesList:
    namePath = dataPath+"/"+i

    for j in os.listdir(namePath):
        labels.append(label)
        facesData.append(cv2.imread(namePath+"/"+j,0))
    
    label+=1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando")
face_recognizer.train(facesData, np.array(labels))

#Almacenando el modelo obtenido
face_recognizer.write('modeloLBPHFace.xml')

#Eso es todo muchas gracias
