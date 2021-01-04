import cv2

# Loading pretrained frontal face data from opencv (HAAR CASCADE ALGO) [https://github.com/opencv/opencv/tree/master/data/haarcascades]
face_detector = cv2.CascadeClassifier(
    'xml-files/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('xml-files/haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('xml-files/haarcascade_eye.xml')

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
            # Drawing a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), ((0, 255, 0)), 2)

            # get the sub frame using numpy N-dimensional arrau slicing
            the_face = frame[y:y+h, x:x+w]

            # converted to grayscale
            grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

            # detect eyes within faces
            eye_coordinates = eye_detector.detectMultiScale(
                grayscaled_face, scaleFactor=1.2, minNeighbors=18)
            for (x_, y_, w_, h_) in eye_coordinates:
                cv2.rectangle(the_face, (x_, y_),
                              (x_ + w_, y_ + h_), ((255, 255, 255)), 2)

            # detect smiles within faces
            smile_coordinates = smile_detector.detectMultiScale(
                grayscaled_face, scaleFactor=1.7, minNeighbors=20)
            for (x_, y_, w_, h_) in smile_coordinates:
                cv2.rectangle(the_face, (x_, y_),
                              (x_ + w_, y_ + h_), ((0, 0, 255)), 2)
            if(len(smile_coordinates) > 0):
                cv2.putText(frame, "Smiling", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((255, 255, 255)), 1)
    else:
        break

    # Show the video box
    cv2.imshow('Smile Detector!!!', frame)
    key = cv2.waitKey(1)

    # Stop on pressing key Q OR q
    if key == 27 or key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
print("End of code")
