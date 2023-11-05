import cv2
import numpy as np
import face_recognition
import os
import math

path = 'Images'
images = []
className = []
myList = os.listdir(path)

for c1 in myList:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    className.append(os.path.splitext(c1)[0])

def findEncoding(images):
    encodeList = []
    for img in images:
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeface, faceloc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        facedis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(facedis)
        facedis=min(facedis)


        if matches[matchIndex]:
            name = className[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            if facedis < 0.48:
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(img, "non_HB", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("webcam", img)
    cv2.waitKey(1)
