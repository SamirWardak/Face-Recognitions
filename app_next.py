from FaceRec import *
from db_manager import database_manager
import streamlit as st
import tempfile

st.title("Face Recognination  Application")

FRAME_WINDOW = st.image([])


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_setup():
    classNames, encoding = set_up()
    return classNames, encoding


classNames, encoding = load_setup()

face_cascade = cv2.CascadeClassifier('face.xml')
cam = cv2.VideoCapture(1)
while True:
    success, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (720, 400), None, .25, .25)

    face = cv2.imread(f'empty.png')
    face = cv2.resize(face, (300, 400), None, .25, .25)

    image = np.hstack((img, face))

    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodings_curret_frame = face_recognition.face_encodings(imgS, faceCurrFrame)

    name = ''

    faces = face_cascade.detectMultiScale(imgS, 1.1, 4)
    if len(faces) > 1:
        name = "multiple faces!!!"
    if len(faces) == 1:
        for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
            matches = face_recognition.compare_faces(encoding, encodeFace)
            face_distance = face_recognition.face_distance(encoding, encodeFace)
            matchIndex = np.argmin(face_distance)
            name = 'Not found found!!!'
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                face = cv2.imread(f'temp_store/{name}.jpg')
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (300, 400), None, .25, .25)
                image = np.hstack((img, face))
                name = 'Name: ' + classNames[matchIndex].upper()
                print(name)

    cv2.putText(image, name, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(image)
    cv2.waitKey(1)