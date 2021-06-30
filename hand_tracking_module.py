import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        # Create a "hands" object
        # Set hand variables to whatever is passed into the class constructor
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_confidence, self.tracking_confidence)

        # Getting drawing utilities for easy hand drawing
        self.mp_draw = mp.solutions.drawing_utils

        # List of tip IDs
        self.tip_ids = [4, 8, 12, 16, 20]

    # Function for detecting hands
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # print(results.multi_hand_landmarks)

        # If hands are detected, loop through them
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Only draw if draw is True(which it is by default)
                if draw:
                    # Drawing landmarks and connections on img
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    # Function for finding the position of a single hand
    def find_position(self, img, hand_number=0, draw=True):
        # List of all x and y values
        x_list = []
        y_list = []
        bounding_box = []

        # List of every detected landmark's id and coordinates
        self.landmark_list = []

        # If hands are detected, loop through them
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]

            # Going through every landmark, with id starting from 0
            for id, landmark in enumerate(my_hand.landmark):
                # print(id, landmark)

                # The coordinates of the landmarks will be returned as ratios of the image
                # This means we need to multiply values by the height and width of the image
                # This is so that we can get precise pixel locations
                height, width, channels = img.shape
                centre_x, centre_y = int(landmark.x * width), int(landmark.y * height)

                # Adding landmarks coordinates to the lists
                x_list.append(centre_x)
                y_list.append(centre_y)

                # print("ID: " + str(id) + " X: " + str(centre_x) + " Y: " + str(centre_y))

                self.landmark_list.append([id, centre_x, centre_y])

                # if draw:
                #     # Testing by drawing circles on the specified landmarks
                #     # 0 is the bottom of the hand, 4 is the tip of the thumb
                #     if id == 0:
                #         cv2.circle(img, (centre_x, centre_y), 25, (255, 0, 255), cv2.FILLED)
                #     elif id == 4:
                #         cv2.circle(img, (centre_x, centre_y), 25, (255, 0, 255), cv2.FILLED)

            # Getting the minimum and maximum x and y values to create a bounding box for the hand
            x_minimum, x_maximum = min(x_list), max(x_list)
            y_minimum, y_maximum = min(y_list), max(y_list)
            bounding_box = x_minimum, y_minimum, x_maximum, y_maximum

        # Return list of landmarks and the bounding box
        return self.landmark_list, bounding_box

    def fingers_up(self):
        fingers = []

        # Get the x value of thumb tips and their corresponding lower knuckles
        # Code for the thumb is different since it's not like the other fingers
        # If the tip is to the right of the lower knuckle, we can say the thumb is up
        # Have yet to check for handedness
        if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            # Get the y value of fingertips and their corresponding lower knuckles
            # If the tip is above the lower knuckle, we can say it's up
            # Then append it to the fingers list
            if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


# To use the code, copy everything in the main function and import the necessary things
def main():
    # Time variables for calculating FPS
    prev_time = 0
    current_time = 0

    # Setting up the webcam
    cap = cv2.VideoCapture(0)

    # Creating a hand detector instance
    detector = HandDetector()

    # Continually show webcam frames
    while True:
        success, img = cap.read()

        # Using find_hands function
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        # Testing the landmark list is working
        if len(landmark_list) != 0:
            print(landmark_list[0])

        # Calculating the FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Drawing the FPS onto img
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey(1)

        # Calculating the FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Drawing the FPS onto img
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
