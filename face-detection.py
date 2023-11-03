from flask import Flask, render_template, request, redirect
import cv2
import face_recognition

import cv2
import numpy as np
import face_recognition
import os, dlib
from datetime import datetime
# from keras.models import load_model
# from PIL import ImageGrab
# from spreadsheet import write_to_sheet, enroll_person_to_sheet, email_exists, mark_absent

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# emotion_model = load_model('model_file.h5')
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Initialize face and eye detectors
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

path = 'uploads'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

EYE_AR_THRESHOLD = 0.25  # Adjust this value as needed
EYE_AR_CONSEC_FRAMES = 3  # Adjust this value as needed


# Helper function to calculate EAR
def calculate_ear(eye_pts):
    A = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
    B = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
    C = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear


path = 'uploads'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name, emotion):
    write_to_sheet(name, emotion)


encodeListKnown = findEncodings(images)
print('Encoding Complete')


@app.route('/')
def index():
    '''now_=datetime.now()
    date_=now_.strftime('%d/%m/%Y')
    print(date_)
    print(now_.strftime('%H:%M:%S'))
    mark_absent(date_,now_.strftime('%H:%M:%S'))'''
    return render_template('index_face.html', name='')


'''
@app.route('/Attendance', methods=['POST'])
def after_register():
    return render_template('index_face.html',name='')'''


@app.route('/attandance', methods=['POST'])
def attandance():
    entered_email = request.form.get('email')
    if email_exists(entered_email) == False:
        print("email not found")
    else:
        name = None
        blink_counter = 0
        frame_counter = 0

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = gray[y:y + h, x:x + w]

                # Resize the face ROI to match the input size of the emotion model (48x48)
                face_roi = cv2.resize(face_roi, (48, 48))

                # Normalize pixel values to be in the range [0, 1]
                face_roi = face_roi / 255.0

                # Expand the dimensions to match the model's input shape (1, 48, 48, 1)
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = np.expand_dims(face_roi, axis=-1)

                # Predict the emotion using the loaded emotion model
                ''' emotion_pred = emotion_model.predict(face_roi)
                emotion_index = np.argmax(emotion_pred)

                # Get the corresponding emotion label
                emotion_label = emotions[emotion_index]'''

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the predicted emotion label above the face
                # cv2.putText(frame, 'emotion_label', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Detect faces using dlib for eye blink detection
            faces = face_detector(gray)

            for face in faces:
                # Get facial landmarks
                landmarks = shape_predictor(gray, face)

                # Extract the coordinates of the eye landmarks
                left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                # Calculate the eye aspect ratio (EAR) for both eyes
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)

                # Calculate the average EAR for both eyes
                ear = (left_ear + right_ear) / 2.0

                # Check if a blink has occurred
                if ear < EYE_AR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                        frame_counter = 0
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)
                faceDis = min(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex]

                # Display the blink count on the frame
                cv2.putText(frame, f'Blinks: {blink_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if blink_counter > 1:
                markAttendance(name, 'emotion_label')
                print(name)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                name = None
                blink_counter = 0
                frame_counter = 0

            # Display the frame with detected emotions and blink count
            cv2.imshow('FRS', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return ""


if __name__ == '_main_':
    app.run(debug=True)