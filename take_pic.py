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
cam = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib-models/shape_predictor_68_face_landmarks_GTX.dat")

def to_rect(r):
    tl = r.tl_corner()
    br = r.br_corner()
    return [(tl.x, tl.y), (br.x, br.y)]


eye_size = 40
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


frame = 0
while True:
    check, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    if rects:
        rect = to_rect(rects[0])
        cv2.rectangle(img, rect[0], rect[1], (255, 0, 0), 2)

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        if shape.size > 0:
            leye = get_leye(shape)
            reye = get_reye(shape)

            cv2.rectangle(img, leye[0], leye[1], (255, 0, 0), 2)
            cv2.rectangle(img, reye[0], reye[1], (255, 0, 0), 2)

            #gray = cv2.GaussianBlur(gray, (3, 3), 0.4)
            leye_img = crop_from_rect(gray, leye)
            print(leye_img.shape)
            leye_img = cv2.resize(leye_img, dsize=(96, 96))
            #leye_img = clahe.apply(leye_img)

            cv2.imshow("leye", leye_img)
            #from datetime import datetime
            #cv2.imwrite(f"out/leye-{datetime.now().strftime('%s')}.png", leye_img)

        face = crop_from_rect(img, rect)
        cv2.imshow("face", face)


    cv2.imshow("img", img)
    frame += 1

    key = cv2.waitKey(5)
    if key == 115:
        if leye_img is not None:
            from datetime import datetime
            print("saving")
            cv2.imwrite(f"own-images/open/leye-{datetime.now().strftime('%H-%M-%S')}.png", leye_img)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
