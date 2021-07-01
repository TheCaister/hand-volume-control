import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math

# Importing pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
minimum_volume = volume_range[0]
maximum_volume = volume_range[1]
vol = 0
volume_bar = 400
volume_percentage = 0

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

    landmarks_list, bounding_box = detector.find_position(img, draw=False)

    # Only process if there is something in the landmark list
    if len(landmarks_list) != 0:
        print(landmarks_list)

        # Getting the area of the bounding box
        bounding_box_area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]) // 100

        # You can only change the volume when the hand is a certain size in the screen
        if 250 < bounding_box_area < 1000:
            # Getting length between two points and info about the line such as end points and centre point
            length, img, line_info = detector.find_distance(4, 8, img)

            # Also converting to range of height of the volume bar
            volume_bar = np.interp(length, [50, 300], [400, 150])
            # And converting to percentage
            volume_percentage = np.interp(length, [50, 300], [0, 100])

            # We can send decimal versions of percentages here (Values between 0 and 1)
            volume.SetMasterVolumeLevelScalar(volume_percentage / 100, None)

            if length < 50:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)

    # Drawing a rectangle representing the volume
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    # Drawing percentage
    cv2.putText(img, f'{int(volume_percentage)}', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)


    # Getting the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    # Drawing the FPS
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)