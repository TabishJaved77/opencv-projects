import cv2

# Loading pretrained data from opencv (HAAR CASCADE ALGO)
bus_tracker = cv2.CascadeClassifier('xml-files/bus_front.xml')
car_tracker = cv2.CascadeClassifier('xml-files/cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('xml-files/pedestrians.xml')
two_wheeler_tracker = cv2.CascadeClassifier('xml-files/two_wheeler.xml')

# choose an image to detect cars in
img = cv2.imread('test-files/car.png')

# converted to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detects traffic
buses = bus_tracker.detectMultiScale(grayscaled_img)
cars = car_tracker.detectMultiScale(grayscaled_img)
pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_img)
two_wheelers = two_wheeler_tracker.detectMultiScale(grayscaled_img)

for (x, y, w, h) in buses:
    cv2.rectangle(img, (x, y), (x+w, y+h), ((255, 0, 0)), 1)
    cv2.putText(img, "Bus", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((255, 0, 0)), 1)
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), ((0, 0, 255)), 1)
    cv2.putText(img, "Car", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ((0, 0, 255)), 1)
for (x, y, w, h) in pedestrians:
    cv2.rectangle(img, (x, y), (x+w, y+h), ((0, 255, 0)), 1)
    cv2.putText(img, "Pedestrian", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ((0, 255, 0)), 1)
for (x, y, w, h) in two_wheelers:
    cv2.rectangle(img, (x, y), (x+w, y+h), ((255, 255, 0)), 1)
    cv2.putText(img, "2-Wheeler", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ((255, 255, 0)), 1)

# Show the video box
cv2.imshow('Traffic Tracker!!!', img)
cv2.waitKey()

print("End of code")
