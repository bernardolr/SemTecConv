import cv2
import os
import imutils


name = "Carlos"
dataPath = "data"
namePath = dataPath +"/"+name

if not os.path.exists(namePath):
    os.makedirs(namePath)

video = cv2.VideoCapture('video2Carlos.mp4')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = video.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.circle(frame,(x+int(w/2),y+int(h/2)),130,(0,0,0),2)
        rostro = aux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(namePath+'/rostro_{}.jpg'.format(count), rostro)
        count +=1
    cv2.imshow('frame',frame)

    k=cv2.waitKey(1)
    if k== 27 or count >= 300:
        break

video.release()
cv2.destroyAllWindows()