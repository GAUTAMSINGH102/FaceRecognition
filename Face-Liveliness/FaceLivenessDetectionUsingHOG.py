import cv2
import numpy as np
import dlib
import imutils
from imutils.video import VideoStream
import tensorflow as tf
import pickle

detector = dlib.get_frontal_face_detector()
le_path = 'label_encoder.pickle'
model = 'liveness.model'
liveness_model = tf.keras.models.load_model(model)
le = pickle.loads(open(le_path, 'rb').read())

#Read the input from video
cap = VideoStream(src=0).start()

while True:
    frame = cap.read()

    frame = imutils.resize(frame, width=600)

    # (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # step3: get HOG face detector and faces
    hogFaceDetector = dlib.get_frontal_face_detector()
    faces = hogFaceDetector(gray, 2)

    # step4: loop through each face and draw a rect around it
    for (i, rect) in enumerate(faces):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        face = frame[y:(y+h), x:(x+w)]

        try:
            face = cv2.resize(face, (32, 32))
        except:
            break

        face = face.astype('float') / 255.0
        face = tf.keras.preprocessing.image.img_to_array(face)

        face = np.expand_dims(face, axis=0)

        preds = liveness_model.predict(face)[0]
        j = np.argmax(preds)
        label = le.classes_[j]  # get label of predicted class

        # draw the label and bounding box on the frame
        label = f'{label}: {preds[j]:.4f}'
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(face.shape)

        # draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output
    # frameS = cv2.resize(frame, (960, 540))
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cv2.destroyAllWindows()
cap.stop()