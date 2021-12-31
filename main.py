import cv2
import math
import numpy as np
from imutils import face_utils
import dlib
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
#import onnx
#import onnxruntime

model = EfficientNet.from_pretrained("efficientnet-b0", in_channels=1)
in_features = model._fc.in_features
model._fc = nn.Sequential(
    nn.Linear(in_features, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Sigmoid()
)
#model_path = "bce-12-31-22-31-26-e5-a95-ta99-effnetb3-drowsiness.pth.tar"
model_path = "bce-12-31-22-42-14-e1-a100-ta98-effnetb3-drowsiness.pth.tar"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"])
model.eval()

cam = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib-models/shape_predictor_68_face_landmarks_GTX.dat")
#predictor = dlib.shape_predictor("dlib-models/shape_predictor_5_face_landmarks.dat")
eye_cascade = cv2.CascadeClassifier()
eye_cascade.load("haarcascade_eye.xml")

def to_rect(r):
    tl = r.tl_corner()
    br = r.br_corner()
    return [(tl.x, tl.y), (br.x, br.y)]


eye_size = 30
def get_leye(shape):
    lx = shape[37][0]
    ly = shape[37][1]
    rx = shape[40][0]
    ry = shape[40][1]
    diff = rx+eye_size - (lx-eye_size)
    return [(lx-eye_size, ly - eye_size), (rx+eye_size, ly-eye_size + diff)]


def get_reye(shape):
    lx = shape[43][0]
    ly = shape[43][1]
    rx = shape[46][0]
    ry = shape[46][1]
    diff = rx+eye_size - (lx-eye_size)
    return [(lx-eye_size, ly - eye_size), (rx+eye_size, ly-eye_size + diff)]


def crop_from_rect(img, rect):
    return img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]


def run_eye(gray, eye):
    eye_img = crop_from_rect(gray, eye)
    eye_img = cv2.resize(eye_img, dsize=(96, 96))
    eye_img = eye_img[np.newaxis, np.newaxis, ...]
    eye_img = (2/255 * eye_img - 1)*3

    with torch.no_grad():
        out = model(torch.tensor(eye_img, dtype=torch.float)).flatten()[0]
        print(out)
    return out, eye_img


frame = 0
while True:
    check, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    eyes = eye_cascade.detectMultiScale(gray)
    for eye in eyes:
        x = eye[0]
        y = eye[1]
        w = eye[2]
        h = eye[3]
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    if rects:
        rect = to_rect(rects[0])
        cv2.rectangle(img, rect[0], rect[1], (255, 0, 0), 2)

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        #for s in shape:
        #    img = cv2.circle(img, [s[0], s[1]], 10, (0, 255, 0))
        if shape.size > 0:
            gray = cv2.GaussianBlur(gray, (3, 3), 0.4)

            leye = get_leye(shape)
            reye = get_reye(shape)

            cv2.rectangle(img, leye[0], leye[1], (255, 0, 0), 2)
            cv2.rectangle(img, reye[0], reye[1], (255, 0, 0), 2)

            leye_out, leye_img = run_eye(gray, leye)
            reye_out, reye_img = run_eye(gray, reye)
            img = cv2.putText(img, f"0 - closed, 100 - open", (100, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                              color=(0, 255, 255), thickness=7)
            img = cv2.putText(img, f"right: {leye_out.numpy()*100:.2f}", (100, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                              color=(0, 255, 255), thickness=5)
            img = cv2.putText(img, f"left: {reye_out.numpy()*100:.2f}", (100, 200), cv2.FONT_HERSHEY_PLAIN, 2,
                              color=(0, 255, 255), thickness=5)

        face = crop_from_rect(img, rect)
        cv2.imshow("face", face)


    cv2.imshow("img", img)
    frame += 1

    key = cv2.waitKey(5)
    if key == 115:
        if leye_img is not None:
            from datetime import datetime
            cv2.imwrite(f"own-images/open/leye-{datetime.now().strftime('%H-%M-%S')}.png", leye_img)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
