import cv2

# Loading pretrained data from opencv (HAAR CASCADE ALGO)
bus_tracker = cv2.CascadeClassifier('xml-files/bus_front.xml')
car_tracker = cv2.CascadeClassifier('xml-files/cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('xml-files/pedestrians.xml')
two_wheeler_tracker = cv2.CascadeClassifier('xml-files/two_wheeler.xml')

# Capturing video footage (can be from a camera too)
webcam = cv2.VideoCapture('test-files/video1.avi')

# Iterate forever over frames
while True:
    # Read the current frame
    read_successful, frame = webcam.read()
    if(read_successful):
        # converted to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detects traffic
        buses = bus_tracker.detectMultiScale(grayscaled_frame)
        cars = car_tracker.detectMultiScale(grayscaled_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
        two_wheelers = two_wheeler_tracker.detectMultiScale(grayscaled_frame)

        for (x, y, w, h) in buses:
            cv2.rectangle(frame, (x, y), (x+w, y+h), ((255, 0, 0)), 1)
            cv2.putText(frame, "Bus", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((255, 0, 0)), 1)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), ((0, 0, 255)), 1)
            cv2.putText(frame, "Car", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((0, 0, 255)), 1)
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), ((0, 255, 0)), 1)
            cv2.putText(frame, "Pedestrian", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ((0, 255, 0)), 1)
        for (x, y, w, h) in two_wheelers:
            cv2.rectangle(frame, (x, y), (x+w, y+h), ((255, 255, 0)), 1)
            cv2.putText(frame, "2-Wheeler", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ((255, 255, 0)), 1)
    else:
        break

    # Show the video box
    cv2.imshow('Traffic Tracker!!!', frame)
    key = cv2.waitKey(1)

    # Stop on pressing key Q OR q
    if key == 27 or key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
print("End of code")
