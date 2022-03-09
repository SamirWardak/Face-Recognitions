from FaceRec import *
from db_manager import database_manager
import streamlit as st
import tempfile
import streamlit_authenticator as stauth


st.title("Face Recognination  Application")
names = ['Ahmad Samir Wardak']
usernames = ['admin']
passwords = ['admin']

hashed_passwords = stauth.hasher(passwords).generate()


authenticator = stauth.authenticate(
    names,
    usernames,
    hashed_passwords,
    'some_cookie_name',
    'some_signature_key',
    cookie_expiry_days=30
)

name, authentication_status = authenticator.login('Login', 'main')

if authentication_status:
    button = st.button('Detect')
    button_clear = st.button('Clear')
    FRAME_WINDOW = st.image([])
    FRAME_WINDOW_DETECT = st.image([])
    name_detected = ''

    def detecters(image, result, face_empty):
        FRAME_WINDOW_DETECT = st.image([])
        img = image
        name_detected = '1'
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(imgS)
        encodings_curret_frame = face_recognition.face_encodings(imgS, faceCurrFrame)

        x, y, w, h = result
        roi_color = img[y:y + h, x:x + w]
        res = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite('result.png', res)
        source_image = cv2.imread(f'result.png')
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_image = cv2.resize(source_image, (360, 400), None, .25, .25)

        frame = np.hstack((source_image, face_empty))

        for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
            matches = face_recognition.compare_faces(encoding, encodeFace)
            face_distance = face_recognition.face_distance(encoding, encodeFace)
            matchIndex = np.argmin(face_distance)
            name_detected = 'Not found found!!!'
            if matches[matchIndex]:
                name_detected = classNames[matchIndex].upper()
                face = cv2.imread(f'temp_store/{classNames[matchIndex]}.jpg')
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (360, 400), None, .25, .25)
                name_detected = 'Name: ' + classNames[matchIndex].upper()
                frame = np.hstack((source_image, face))
            FRAME_WINDOW_DETECT.image(frame)
        return name_detected


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

        face_empty = cv2.imread(f'empty.png')
        face_empty = cv2.resize(face_empty, (300, 400), None, .25, .25)
        faces = face_cascade.detectMultiScale(img, 1.1, 4)


        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if button:
                name_detected = detecters(img, faces[0], face_empty)
        if button_clear:
            name_detected = ''
            FRAME_WINDOW_DETECT = st.empty()

        button = False
        cv2.putText(img, name_detected, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(img)
        cv2.waitKey(1)
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')