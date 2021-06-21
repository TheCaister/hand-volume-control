import cv2
import time
import numpy as np

########################
camera_width, camera_height = 640, 480
########################

# Setting up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

previous_time = 0

while True:
    success, img = cap.read()

    # Getting the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)