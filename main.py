import cv2
import math
import numpy as np
from imutils import face_utils
import dlib

import torch
import onnx
import onnxruntime

cam = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib-models/shape_predictor_68_face_landmarks_GTX.dat")

drowsiness_model = onnx.load("drowsiness.onnx")
onnx.checker.check_model(drowsiness_model)
ort_session = onnxruntime.InferenceSession("drowsiness.onnx", None)


def to_rect(r):
    tl = r.tl_corner()
    br = r.br_corner()
    return [(tl.x, tl.y), (br.x, br.y)]


def get_leye(shape):
    lx = shape[37][0]
    ly = shape[37][1]
    rx = shape[40][0]
    ry = shape[40][1]
    diff = rx+30 - (lx-30)
    return [(lx-30, ly - 30), (rx+30, ly-30 + diff)]


def get_reye(shape):
    lx = shape[43][0]
    ly = shape[43][1]
    rx = shape[46][0]
    ry = shape[46][1]
    diff = rx+30 - (lx-30)
    return [(lx-30, ly - 30), (rx+30, ly-30 + diff)]


def crop_from_rect(img, rect):
    return img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]


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

            leye_img = crop_from_rect(gray, leye)
            leye_img = cv2.resize(leye_img, dsize=(224, 224))

            leye_img = clahe.apply(leye_img)

            cv2.imshow("leye", leye_img)
            #cv2.imwrite("leye.png", leye_img)
            leye_img = np.stack((leye_img, leye_img, leye_img), axis=0)[np.newaxis, ...]
            print(leye_img.shape)

            ortinput = onnxruntime.OrtValue.ortvalue_from_numpy(leye_img.astype(np.single)/255)
            ort_inputs = {"input": ortinput}
            ort_outs = ort_session.run_with_ort_values(["output"], ort_inputs)[0].numpy()[0]
            total = math.exp(ort_outs[0]) + math.exp(ort_outs[1])
            ort_outs[0] = math.exp(ort_outs[0])/total
            ort_outs[1] = math.exp(ort_outs[1])/total
            print("%.3f %.3f" % (ort_outs[0] * 100, ort_outs[1] * 100))

        face = crop_from_rect(img, rect)
        cv2.imshow("face", face)


    cv2.imshow("img", img)

    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
