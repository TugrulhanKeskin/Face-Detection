import cv2
import numpy as np

# Yüz tanımlama için cascade classifier'ı yükle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kamera'ya eriş
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü oku
    ret, frame = cap.read()
    # yüzleri tespit et
    if ret : 
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=5)
        # yüzleri çerçevele
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
        cv2.imshow("face detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Kamerayı serbest bırak
cap.release()
cv2.destroyAllWindows()