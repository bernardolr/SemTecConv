import cv2
import os

dataPath = "data"
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')

video = cv2.VideoCapture("video6Carlos.mp4")
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


while True:
    ret,frame = video.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux = gray.copy()
faces = faceClassif.detectMultiScale(gray,1.3,5)
