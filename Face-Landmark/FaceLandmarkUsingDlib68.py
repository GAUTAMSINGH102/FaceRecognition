import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
images = 'images'

# Read the Input Image
img = cv2.imread('./images/group-people.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray, 2)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    print(face.top())
    print(face.left())
    print(face.right())
    print(face.bottom())

    landmarks = predictor(gray, face)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, center=(x, y), radius=1, color=(255, 0, 0), thickness=-1)

# Display the output
imS = cv2.resize(img, (960, 540))
cv2.imshow('img', imS)
cv2.waitKey()


# # Read the Input Video
# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = detector(gray)
#
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()
#         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#         landmarks = predictor(gray, face)
#
#         for n in range(0, 68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
#             cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
#
#
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break