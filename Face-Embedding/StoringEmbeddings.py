import cv2
import numpy as np
import dlib
import os
import pickle
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("../Models/dlib_face_recognition_resnet_model_v1.dat")
directory = 'delete'

list_of_face_embedding = []
list_of_names = []
face_embedding = pd.DataFrame()

for filename in os.listdir(directory):
    name = filename.split('.')[-2]
    images = os.path.join(directory, filename)

    # Read the Input Image
    img = cv2.imread(images)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    embedding = []

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, center=(x, y), radius=3, color=(255, 0, 0), thickness=-1)

        face_embedding = np.array(face_encoder.compute_face_descriptor(img, landmarks, num_jitters=1))
        list_of_face_embedding.append(face_embedding)
        list_of_names.append(name)
        embedding.append(face_embedding)
    print("******************************************")
    print(name)
    print(images)
    print(len(embedding))
    print(len(list_of_face_embedding))
    print("*******************************************")

    # Display the output
    imS = cv2.resize(img, (960, 540))
    cv2.imshow('img', imS)
    cv2.waitKey()

print(len(list_of_face_embedding))
print(len(list_of_names))

list_of_tuples = list(zip(list_of_names, list_of_face_embedding))

Embedding = pd.DataFrame(list_of_tuples, columns=['Name', 'Embedding'])

pickle.dump(Embedding, open('face_embeddings.pkl', 'wb'))



