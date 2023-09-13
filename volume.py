#Libraries
import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

np.set_printoptions(precision=3)

#Mediapipe Hands
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
LMS = mpHands.HandLandmark

#Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#Camera
capture = cv2.VideoCapture(0)

hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=1)

while capture.isOpened():
    success, image = capture.read()

    if not success:
        continue

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processedHands = hands.process(imageRGB)

    #Check and Draw hand landmarks
    if processedHands.multi_hand_landmarks is not None:
        for handLandmarks in processedHands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)

            lm = handLandmarks.landmark
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
            #Change volume based on distance between index and thumb
            VOLUME_SPEED = 0.1
            diff = np.interp(length, (0.0, maxDistance), [-VOLUME_SPEED, VOLUME_SPEED])

            currentVolume = volume.GetMasterVolumeLevelScalar()
            newVolume = min(max((currentVolume + diff), 0.0), 1.0)
            volume.SetMasterVolumeLevelScalar(newVolume, None)
            
    #Show image
    cv2.imshow('Volume', image)

    #press q to quit
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
        
#Release camera
capture.release()
cv2.destroyAllWindows()