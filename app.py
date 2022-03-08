from FaceRec import *
from db_manager import database_manager
import streamlit as st
import tempfile

st.title("Face Recognination  Application")

source_of_content = st.sidebar.selectbox('The the Source of Input', options=['<select>',  'Image', 'Video'])
add_new_face = st.sidebar.checkbox('Add New Face')
FRAME_WINDOW = st.image([])


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_setup():
    classNames, encoding = set_up()
    return classNames, encoding


classNames, encoding = load_setup()

if add_new_face:
    name = st.text_area("Enter Name of the Person")
    img_file_buffer = st.file_uploader("Upload an image you want to add in database", type=["jpg", "jpeg", "png"])
    add_face = st.button('Add Face')
    if name and img_file_buffer and add_face:
        image = img_file_buffer.read()
        database_manager().add_image(name, img=image)
        image_ = face_recognition.load_image_file(img_file_buffer)
        st.text('New Image Added to DataBase')
        encoding_new = single_image_encoding(image_)
        encoding.append(encoding_new)
        classNames.append(name)

if source_of_content == 'WebCam':
    cam = cv2.VideoCapture(1)
    while True:
        success, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(imgS)
        encodings_curret_frame = face_recognition.face_encodings(imgS, faceCurrFrame)

        for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
            matches = face_recognition.compare_faces(encoding, encodeFace)
            face_distance = face_recognition.face_distance(encoding, encodeFace)
            matchIndex = np.argmin(face_distance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                colors = (0, 255, 0)
            else:
                name = 'No Match'
                colors = (0, 0, 255)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), colors, 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), colors, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        FRAME_WINDOW.image(img)
        cv2.waitKey(1)

elif source_of_content == 'Image':
    face_from_image = st.file_uploader("Upload an image to recognize the face", type=["jpg", "jpeg", "png"], key='for Images')
    if face_from_image:
        img = face_recognition.load_image_file(face_from_image)
        faceCurrFrame = face_recognition.face_locations(img)
        encodings_curret_frame = face_recognition.face_encodings(img, faceCurrFrame)
        for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
            matches = face_recognition.compare_faces(encoding, encodeFace)

            face_distance = face_recognition.face_distance(encoding, encodeFace)
            matchIndex = np.argmin(face_distance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                colors = (0, 255, 0)
            else:
                name = 'No Match'
                colors = (0, 0, 255)
            y1, x2, y2, x1 = faceLoc

            cv2.rectangle(img, (x1, y1), (x2, y2), colors, 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), colors, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

            st.image(img, caption=f"detected Image {name}", use_column_width=True)


elif source_of_content == 'Video':
    video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "gif"])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, img = cap.read()
            faceCurrFrame = face_recognition.face_locations(img)
            encodings_curret_frame = face_recognition.face_encodings(img, faceCurrFrame)

            for encodeFace, faceLoc in zip(encodings_curret_frame, faceCurrFrame):
                matches = face_recognition.compare_faces(encoding, encodeFace)

                face_distance = face_recognition.face_distance(encoding, encodeFace)
                matchIndex = np.argmin(face_distance)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    colors = (0, 255, 0)
                else:
                    name = 'No Match'
                    colors = (0, 0, 255)
                y1, x2, y2, x1 = faceLoc

                cv2.rectangle(img, (x1, y1), (x2, y2), colors, 2)
                cv2.rectangle(img, (x1, y2 - 30), (x2, y2), colors, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

                stframe.image(img, channels="BGR", use_column_width=True)
