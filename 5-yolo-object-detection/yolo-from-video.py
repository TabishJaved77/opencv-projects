import cv2
import numpy as np

# loading YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNet('yolo-coco-data/yolov3.weights',
                      'yolo-coco-data/yolov3.cfg')

# loading the COCO class labels the YOLO model was trained on
classes = []
with open('yolo-coco-data/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Capturing live video from camera
# webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('test-files/office.mp4')

# Iterate forever over frames
while True:
    # Read the current frame
    read_successful, frame = webcam.read()
    if(read_successful):
        _, frame = webcam.read()
        height, width, _ = frame.shape

        # constructing a blob from chosen image. Forward pass the YOLO object detector, giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        # initialization lists
        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                # extract class ID and confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filtering out weak predictions
                if confidence > 0.5:  # confidence=0.5; minimum probability to filter weak detections, IoU threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # use the center (x, y)-coordinates to derive the top and and left corner of the box
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    # updating list of box coordinates, confidences, and class IDs
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        # apply non-maxima suppression to suppress weak, overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        """[confidence=0.5; minimum probability to filter weak detections, IoU threshold]
           [threshold =0.4; threshold when applying non-maxima suppression]"""

        # ensure at least one detection exists
        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i]*100, 2))

                # drawing box rectangle and label on the image
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))

                color = colors[i]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label + ": " + confidence + "%",
                            (x, y-5), font, 1, (255, 255, 255), 1)

    else:
        break

    # Show the video box
    cv2.imshow('Object Detector!!!', frame)
    key = cv2.waitKey(1)

    # Stop on pressing key Q OR q
    if key == 27 or key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()

print("**End of code**")
