import cv2

# Loading pretrained frontal face data from opencv (HAAR CASCADE ALGO) [https://github.com/opencv/opencv/tree/master/data/haarcascades]
face_detector = cv2.CascadeClassifier(
    'xml-files/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('xml-files/haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('xml-files/haarcascade_eye.xml')

# choose an image to detect faces in
img = cv2.imread('test-files/smilyface.jpg')

# converted to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = face_detector.detectMultiScale(grayscaled_img)

# Drawing green rectangles around the face area. First-tuple: topleft corner, second-tuple: bottom right corner, color(RGB), rectangle-width
for (x, y, w, h) in face_coordinates:
    # Drawing a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), ((0, 255, 0)), 2)
    # get the sub frame using numpy N-dimensional arrau slicing
    the_face = img[y:y+h, x:x+w]

    # converted to grayscale
    grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

    # detect eyes within faces
    eye_coordinates = eye_detector.detectMultiScale(
        grayscaled_face, scaleFactor=1.1, minNeighbors=20)
    for (x_, y_, w_, h_) in eye_coordinates:
        cv2.rectangle(the_face, (x_, y_),
                      (x_ + w_, y_ + h_), ((255, 255, 255)), 2)

    # detect smiles within faces
    smile_coordinates = smile_detector.detectMultiScale(
        grayscaled_face, scaleFactor=1.3, minNeighbors=20)
    for (x_, y_, w_, h_) in smile_coordinates:
        cv2.rectangle(the_face, (x_, y_),
                      (x_ + w_, y_ + h_), ((0, 0, 255)), 2)
    if(len(smile_coordinates) > 0):
        cv2.putText(img, "Smiling", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((255, 255, 255)), 1)

cv2.imshow('Smile Detector!!!', img)
cv2.waitKey()
print("End of code")
