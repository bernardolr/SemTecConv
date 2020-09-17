import cv2

# cargar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cargar imagen
img = cv2.imread('test.jpg')

# convertir a greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar cara
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# enmarcar cara
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# mostrar imagen
cv2.imshow('img', img)
cv2.waitKey()