import cv2
import os

dataPath = "data"
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('modeloLBPHFace.xml')

video = cv2.VideoCapture("obama.mp4")
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


while True:
    ret,frame = video.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        face = aux[y:y+h,x:x+w]
        face = cv2.resize(face,(150,150),interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(face)

        if result[1] < 70:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x+int(w/2)-70,y+h+60),2,1.1,(56,15,124),1,cv2.LINE_AA)
            cv2.circle(frame,(x+int(w/2),y+int(h/2)),120,(0,255,0),-1)
        else:
            cv2.putText(frame,'Intruso',(x+int(w/2)-70,y+h+60),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.circle(frame,(x+int(w/2),y+int(h/2)),120,(255,0,0),-1)


    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k==27:
        break

video.release()
cv2.destroyAllWindows()
