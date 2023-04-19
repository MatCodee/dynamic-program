'''
# Aqui definidmos el formato de entrada en lo que respecta el video
Definirmos la conversiond e un video a una monton de imagenes
'''
import cv2
import numpy as np


def reading_video():
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    cap.open("video.mp4")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return np.array(frames)


