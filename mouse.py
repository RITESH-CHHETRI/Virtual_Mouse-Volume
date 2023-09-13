#Libraries
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time 

np.set_printoptions(precision=3)

#Mediapipe Hands
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
LMS = mpHands.HandLandmark

#Get window size
screen_width, screen_height = pyautogui.size()

#Camera
wcam = 820
hcam = 1000
capture = cv2.VideoCapture(0)
capture.set(3,wcam)
capture.set(4, hcam)
time.sleep(1)

hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=1)

while capture.isOpened():
    success, image = capture.read()

    if not success:
        continue
    h,w,c=image.shape
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processedHands = hands.process(imageRGB)

    #Check and Draw hand landmarks
    if processedHands.multi_hand_landmarks is not None:
        for handLandmarks in processedHands.multi_hand_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)
            lm = handLandmarks.landmark

            #Get coordinates of index and draw circle
            x,y = int(lm[LMS.INDEX_FINGER_TIP].x*w), int(lm[LMS.INDEX_FINGER_TIP].y*h)
            cv2.circle(image,(x,y),20,(0, 255, 0),cv2.FILLED)

            #Move mouse to index finger tip
            x,y = int(lm[LMS.INDEX_FINGER_TIP].x*screen_width), int(lm[LMS.INDEX_FINGER_TIP].y*screen_height)
            pyautogui.moveTo(screen_width-x,y)

            #Get coordinates
            thumb = np.array([lm[LMS.THUMB_TIP].x,        lm[LMS.THUMB_TIP].y])
            index = np.array([lm[LMS.INDEX_FINGER_TIP].x, lm[LMS.INDEX_FINGER_TIP].y])
            wrist = np.array([lm[LMS.WRIST].x,            lm[LMS.WRIST].y])

            #Length of index to wrist
            indexLength = np.linalg.norm(index - wrist)
            #Length of thumb to wrist
            thumbLength = np.linalg.norm(thumb - wrist)
            #Max distance between index and thumb using pythagoras
            maxDistance = 0.7 * math.hypot(indexLength, thumbLength)

            length = np.linalg.norm(index - thumb)
            #If length is less than 0.05, click
            if length < 0.05:
                pyautogui.click(screen_width-x,y)

    #Flip image to give mirror effect
    flipped = cv2.flip(image, 1)

    #Show image
    cv2.imshow('Mouse', flipped)

    #Press q to quit
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break

capture.release()
cv2.destroyAllWindows()