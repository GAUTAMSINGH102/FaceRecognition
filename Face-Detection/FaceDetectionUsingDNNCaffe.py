import cv2
import numpy as np

#step1:Read model and prototxt file
caffeModelFile = "../Models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
caffeProtoTxt = "../Models/face_detector/deploy.prototxt"

#step2:Create a readNet object from the caffemodel
caffeDetector = cv2.dnn.readNetFromCaffe(caffeProtoTxt, caffeModelFile)

#step3: read the input image
image = cv2.imread("../images/avenger_5.jpg")
(h, w) = image.shape[:2] #get image height and width

#step4: convert image to blob
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300))

#step5: give it to the detector and fetch all faces
caffeDetector.setInput(blob)
detections = caffeDetector.forward()
print(detections.shape)

#step6: go through each face and draw rectangle if face confidence percentage is at least 30%
conf_threshold = 0.3
for faceIndex in range(0, detections.shape[2]):
    confidence = detections[0, 0, faceIndex, 2]
    if confidence > conf_threshold:  #filter the face confidence percentage
        # computer (x, y) coordinates
        (startX, startY, endX, endY) = (detections[0, 0, faceIndex, 3:7]
                                        * np.array([w, h, w, h])).astype("int")

        # draw rectangle
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255,0), 2)

        # draw confidence percentage
        cv2.putText(image, "{:.2f}%".format(confidence * 100),
                    (startX, startY-10),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 2), 2)

#step7: display the image
cv2.imshow("Image", image)
cv2.waitKey(0)