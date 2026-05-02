import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
names = {}
current_id = 0

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    name = file.split("_")[0]

    if name not in names:
        names[name] = current_id
        current_id += 1

    label_id = names[name]

    detected_faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        faces.append(face)
        labels.append(label_id)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

print("Training selesai!")
print(names)