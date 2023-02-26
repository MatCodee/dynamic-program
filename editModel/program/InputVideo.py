'''
# Aqui definidmos el formato de entrada en lo que respecta el video
Definirmos la conversiond e un video a una monton de imagenes
'''
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')


def reading_video():
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    video_array = np.array(frames)
    return video_array

