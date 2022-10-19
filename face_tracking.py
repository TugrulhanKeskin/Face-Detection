import cv2
import numpy as np
import time 

# video aktar 
video = "face.mp4"
cap = cv2.VideoCapture(video)

# Bir tane frame oku
ret, frame = cap.read()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

if ret == False:
    print("video yüklenmedi")
    exit()

# Detection 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

# meanshift algoritması girdisi
track_window = (face_x, face_y, w, h) 

# ROI: Region of Interest : Tespit ettiğimiz kutunun içindeki alan = face
roi = frame[face_y:face_y+h, face_x:face_x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Histogram oluştur
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180]) #Takip için histogram gerekli 
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX) # Normalizasyon

# takip için gerekli durdurma kriterleri
# count = hesaplanacak maksimum iterasyon sayısı 
# eps = iterasyon sonunda maksimum değişim değeri
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

# Video kaydı için
writer = cv2.VideoWriter("video_kaydi.mp4", cv2.VideoWriter_fourcc(*"mp4v"),20, size)

# Takip
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # meanshift algoritması
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1) # ROI histogramı ile frame histogramı piksel karşılaştırılır
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    
    x,y,w,h = track_window
    img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 5)
    
    time.sleep(0.01)
    cv2.imshow("Frame", frame)

    # kaydetme
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()