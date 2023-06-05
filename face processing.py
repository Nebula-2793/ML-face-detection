import cv2
import numpy as np

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector.detectMultiScale(img, 1.3, 5) #returns a list of tuples, each tuple -> (x_coord, y_coord, width, height)
    if face == ():
        return None
    else:
        for (x, y, w, h) in face:
            i = img[y: y+h, x:x+w]
        return i

cap = cv2.VideoCapture(0)

while True:
    status, frame = cap.read()
    txtFrame = cv2.putText(frame, "hello", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("cam", frame)
    face_img = face_extractor(frame)

    if (face_img is not None):
        cv2.imshow("face", face_img)

    if cv2.waitKey(1) == 27:
        break

    
cv2.destroyAllWindows
cap.release()