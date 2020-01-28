import cv2
import os

def detect_faces(video_file, detector, win_title):
    cap = cv2.VideoCapture(video_file)
    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,
            0), 3)
            text_size, _ = cv2.getTextSize('Face',
            cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x, y - text_size[1]), (x +
            text_size[0], y), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, 'Face', (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow(win_title, frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
capture = cv2.VideoCapture(0)

haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_alt.xml')
haar_face_cascade = cv2.CascadeClassifier(haar_model)

detect_faces(0, haar_face_cascade, 'Haar cascade face detector')
