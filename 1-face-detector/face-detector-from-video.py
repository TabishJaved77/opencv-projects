import cv2
from random import randrange

# Loading pretrained frontal face data from opencv (HAAR CASCADE ALGO) [https://github.com/opencv/opencv/tree/master/data/haarcascades]
face_detector = cv2.CascadeClassifier(
    'xml-files/haarcascade_frontalface_default.xml')

# Capturing live video from camera
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('test-files/xyz.mp4')  # from a video file

# Iterate forever over frames
while True:
    # Read the current frame
    read_successful, frame = webcam.read()
    if(read_successful):
        # converted to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        face_coordinates = face_detector.detectMultiScale(grayscaled_frame)
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)
    else:
        break

    # Show the video box
    cv2.imshow('Face Detector!!!', frame)
    key = cv2.waitKey(1)

    # Stop on pressing key Q OR q
    if key == 27 or key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
print("End of code")
