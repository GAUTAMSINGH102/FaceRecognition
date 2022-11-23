#DISADVANTAGE
# The only disadvantage with HOG based face detection is that it doesn’t work on faces at odd angles,
# it only works with straight and front faces. It is really useful if you use it to detect faces from scanned documents
# like driver’s license and passport but not a good fit for real-time video.

# Let’s do coding to detect faces using HOG, dlib library has a straight forward method to return HOG face detector
# “dlib.get_frontal_face_detector()”

import dlib
import cv2

# step1: read the image
image = cv2.imread("../images/avenger_5.jpg")

# step2: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# step3: get HOG face detector and faces
hogFaceDetector = dlib.get_frontal_face_detector()
faces = hogFaceDetector(gray, 2)

# step4: loop through each face and draw a rect around it
for (i, rect) in enumerate(faces):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # draw a rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# step5: display the resulted image
imS = cv2.resize(image, (960, 540))
cv2.imshow("Image", imS)
cv2.waitKey()


# #Read the input from video
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     _, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # step3: get HOG face detector and faces
#     hogFaceDetector = dlib.get_frontal_face_detector()
#     faces = hogFaceDetector(gray)
#
#     # step4: loop through each face and draw a rect around it
#     for (i, rect) in enumerate(faces):
#         x = rect.left()
#         y = rect.top()
#         w = rect.right() - x
#         h = rect.bottom() - y
#         # draw a rectangle
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display the output
#     # frameS = cv2.resize(frame, (960, 540))
#     cv2.imshow('img', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()