from cv2 import cv2
import os
import imutils 

personName = 'Bernardo'
dataPath = "C:/Users/Bernardo/Github/VS/Data"
personPath = dataPath + '/' + personName
#print(personPath)
if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture('VideoBernardo.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_')