import cv2
from random import randrange

# Loading pretrained frontal face data from opencv (HAAR CASCADE ALGO) [https://github.com/opencv/opencv/tree/master/data/haarcascades]
face_detector = cv2.CascadeClassifier(
    'xml-files/haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('test-files/group.png')

# converted to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = face_detector.detectMultiScale(grayscaled_img)
# print(face_coordinates)

# Drawing green rectangles around the face area. First-tuple: topleft corner, second-tuple: bottom right corner, color(RGB), rectangle-width
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),
                                            randrange(128, 256), randrange(128, 256)), 2)

cv2.imshow('Face Detector!!!', img)
cv2.waitKey()

print("End of code")
