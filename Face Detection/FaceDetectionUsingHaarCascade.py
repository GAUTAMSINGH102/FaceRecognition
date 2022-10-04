#DISADVANTAGE
# Haar face detection doesn't work when the face is rotated or at a different angle
# It gives a lot of false predictions

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('./images/avenger_5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y , w ,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

# Display the output
imS = cv2.resize(img, (960, 540))
cv2.imshow('img', imS)
cv2.waitKey()

# #Read the input from video
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     _, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
#     for (x, y , w ,h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0 , 0), 3)
#
#     # Display the output
#     frameS = cv2.resize(frame, (960, 540))
#     cv2.imshow('img', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()