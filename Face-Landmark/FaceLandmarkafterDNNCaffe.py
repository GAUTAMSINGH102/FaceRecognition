import cv2
import numpy as np
import dlib

predictor = dlib.shape_predictor("../Models/shape_predictor_68_face_landmarks.dat")

#step1:Read model and prototxt file
caffeModelFile = "../Models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
caffeProtoTxt = "../Models/face_detector/deploy.prototxt"

#step2:Create a readNet object from the caffemodel
caffeDetector = cv2.dnn.readNetFromCaffe(caffeProtoTxt, caffeModelFile)

# Read the Input Image
img = cv2.imread('../images/group-people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(h, w) = img.shape[:2] #get image height and width

#step4: convert image to blob
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300))

#step5: give it to the detector and fetch all faces
caffeDetector.setInput(blob)
detections = caffeDetector.forward()

#step6: go through each face and draw rectangle if face confidence percentage is at least 30%
conf_threshold = 0.3
for faceIndex in range(0, detections.shape[2]):
    confidence = detections[0, 0, faceIndex, 2]
    if confidence > conf_threshold:  #filter the face confidence percentage
        # computer (x, y) coordinates
        (startX, startY, endX, endY) = (detections[0, 0, faceIndex, 3:7]
                                        * np.array([w, h, w, h])).astype("int")

        faceRect = dlib.rectangle(startX, startY, endX, endY)
        landmarks = predictor(gray, faceRect)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, center=(x, y), radius=1, color=(255, 0, 0), thickness=-1)

# Display the output
imS = cv2.resize(img, (960, 540))
cv2.imshow('img', imS)
cv2.waitKey()