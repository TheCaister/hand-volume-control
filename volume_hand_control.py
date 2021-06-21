import cv2
import time
import numpy as np
import hand_tracking_module as htm

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
    # Drawing the FPS
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)