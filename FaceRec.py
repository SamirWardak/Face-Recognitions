from base64 import encode
import cv2
import numpy as np
import face_recognition
import os
from db_manager import database_manager


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def single_image_encoding(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    return encode


def set_up():
    path = 'temp_store'
    if not os.path.exists(path):
        os.makedirs(path)

    database_manager().write(path)

    images = []

    classNames = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print('Creating Encodings...')
    encoding = find_encodings(images)
    print('Encodings Created')

    # for f in os.listdir(path):
    #     os.remove(os.path.join(path, f))

    return classNames, encoding
