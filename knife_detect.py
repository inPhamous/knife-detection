import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np

def prepare_image(frame):
    IMG_SIZE = 200
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [cv2.resize(gray, (IMG_SIZE, IMG_SIZE))]


def detect_knives(video_file, detector, win_title):
    cap = cv2.VideoCapture(video_file)
    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break
        knives = detector.predict(prepare_image(frame))

        print(knives)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
capture = cv2.VideoCapture(0)

model = load_model("model", compile=False);

detect_knives(0, model, 'KnifeDetect')
