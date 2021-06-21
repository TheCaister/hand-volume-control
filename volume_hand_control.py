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

# Making detector object, bumping up detection confidence the minimise flicker
detector = htm.HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()

    # Find hands and draw landmarks on img
    img = detector.find_hands(img)

    landmarks_list = detector.find_position(img, draw=False)

    # Only process if there is something in the landmark list
    if len(landmarks_list) != 0:
        print(landmarks_list)

        # Getting positions of the index and thumb tips
        x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)


# Getting the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    # Drawing the FPS
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)