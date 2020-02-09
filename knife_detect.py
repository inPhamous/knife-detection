import cv2
import os
from tensorflow.keras.models import model_from_json 
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

        predict = model.predict(prepare_image(frame).reshape(1, 200, 200, 1))
        
        print(predict)
        cv2.imshow(win_title, frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

# Loading the model
json_file = open("./models/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("./models/model-bw.h5")
print("Loaded model from disk")

detect_knives(0, loaded_model, 'Knife Detect')
