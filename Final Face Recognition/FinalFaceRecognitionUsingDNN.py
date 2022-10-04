import cv2
import numpy as np
import dlib
import pickle
import imutils
from imutils.video import VideoStream
import tensorflow as tf

# For face detection
caffeModelFile = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
caffeProtoTxt = "./face_detector/deploy.prototxt"
caffeDetector = cv2.dnn.readNetFromCaffe(caffeProtoTxt, caffeModelFile)
# For face landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# For face embedding
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# For Liveliness Detection
model = 'liveness.model'
liveness_model = tf.keras.models.load_model(model)
le_path = 'label_encoder.pickle'
le = pickle.loads(open(le_path, 'rb').read())

confidencearg = 0.5

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    compare = []
    # return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    torf = (face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    dis = np.round(face_distance(known_face_encodings, face_encoding_to_check), 2)

    for i in zip(torf, dis):
        compare.append(i)
    # zipped = zip(torf, dis)
    # compare.append(zipped)
    return compare

def face_distance(face_encodings, face_to_compare):
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

known_embedding = pickle.load(open('face_embeddings.pkl', 'rb'))
list_know_embedding = []
for embedding in known_embedding['Embedding']:
    list_know_embedding.append(embedding)

# # Read the Input Image
# img = cv2.imread('./images/avenger_5.jpg')

#Read the input from video
cap = VideoStream(src=0).start()

while True:
    frame = cap.read()
    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    caffeDetector.setInput(blob)
    detections = caffeDetector.forward()
    print(detections.shape)

    # iterate over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > confidencearg:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            # print(f"box is {box}")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            livelinessFace = frame[y1:y2, x1:x2]
            face = frame[y1:y2, x1:x2]
            # rgbFace = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            try:
                livelinessFace = cv2.resize(livelinessFace, (32, 32))
            except:
                break

            livelinessFace = livelinessFace.astype('float') / 255.0
            livelinessFace = tf.keras.preprocessing.image.img_to_array(livelinessFace)
            livelinessFace = np.expand_dims(livelinessFace, axis=0)

            preds = liveness_model.predict(livelinessFace)[0]
            j = np.argmax(preds)
            label = le.classes_[j]  # get label of predicted class
            print(label)
            labelWithPred = f'{label}: {preds[j]:.4f}'

            if (label == 'real'):
                faceRect = dlib.rectangle(x1, y1, x2, y2)
                landmarks = predictor(gray, faceRect)

                # for n in range(0, 68):
                #     x = landmarks.part(n).x
                #     y = landmarks.part(n).y
                #     cv2.circle(img, center=(x, y), radius=1, color=(255, 0, 0), thickness=-1)

                main_face_embedding = np.array(face_encoder.compute_face_descriptor(frame, landmarks, num_jitters=1))

                listof = compare_faces(list_know_embedding, main_face_embedding)
                print(listof)

                torf = []
                for idx, info in enumerate(listof):
                    torf.append(info[0])

                confirmation = True
                for idx, info in enumerate(listof):
                    if confirmation in torf:
                        if (info[0] == True):
                            name = known_embedding['Name'][idx]
                            distanc = str(info[1])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                            cv2.putText(frame, distanc, (x2 - 20, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                        2)
                            cv2.putText(frame, label, (x2 - 20, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            break

                    else:
                        name = "Unknown"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        cv2.putText(frame, label, (x2 - 20, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # confirmation = True
    # for idx, info in enumerate(listof):
    #     if confirmation in listof:
    #         if (info == True):
    #             name = known_embedding['Name'][idx]
    #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    #             cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    #             break
    #
    #     else:
    #         name = "Unknown"
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #         cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # try:
    #     idx = listof.index(True)
    #     name = known_embedding['Name'][idx]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    #     cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    # except:
    #     name = "Unknown"
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #     cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Display the output
# imGS = cv2.resize(group_img, (960, 540))
# cv2.imshow('img', imGS)
# cv2.waitKey()




