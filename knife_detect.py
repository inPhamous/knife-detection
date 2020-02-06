import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

def prepare_image(frame):
    IMG_SIZE = 200
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

def detect_knives(video_file, model,  win_title):
    cap = cv2.VideoCapture(video_file)
    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break
        net = cv2.dnn.readNet(model)

        imgGray = prepare_image(frame)
        imgGray = imgGray.astype(np.uint8)
        blob = cv2.dnn.blobFromImage(imgGray,size= (200,200), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()
        print(output)
        
        # cvNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_graph.pb');
        # cvNet.setInput(cv2.dnn.blobFromImage(prepare_image(frame), size=(200, 200), swapRB=True, crop=False))

        cv2.imshow(win_title, imgGray)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

model = './frozen_models/frozen_graph.pb'

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
capture = cv2.VideoCapture(0)

detect_knives(0, model, 'Knife Detect')
