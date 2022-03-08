from FaceRec import *


def load_setup():
    classNames, encoding = set_up()
    return classNames, encoding


classNames, encoding = load_setup()

face_cascade = cv2.CascadeClassifier('face.xml')
cam = cv2.VideoCapture(1)

while True:
    success, img = cam.read()
    final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(final, 1.1, 4)
    text = 'Not Found Face!!!'
    if len(faces) == 0:
        # cv2.imshow('Not Found Face!!!', img)
        pass
    elif len(faces) == 1:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(imgS)
        encodings_curret_frame = face_recognition.face_encodings(imgS, faceCurrFrame)
        name = 'No Match in database'

        for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
            matches = face_recognition.compare_faces(encoding, encodeFace)
            face_distance = face_recognition.face_distance(encoding, encodeFace)
            matchIndex = np.argmin(face_distance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
        text = 'Face detected: '+str(name)+'.'

    else:
        text = 'Allow only one face!!!'
        pass
    cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("Main", img)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
