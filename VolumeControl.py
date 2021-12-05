import cv2
import mediapipe as mp
import numpy as np
import time
import math

import HandTracker

volume = 0

pTime = 0
cap = cv2.VideoCapture(0)

detector = HandTracker.HandDetector()

while True:
    success, img = cap.read()
    img = detector.drawLandmarksToImage(img)

    thumbTipPos = detector.getLandmarkPosition(img, 0, HandTracker.HandLandmark.THUMB_TIP)
    indexTipPos = detector.getLandmarkPosition(img, 0, HandTracker.HandLandmark.INDEX_FINGER_TIP)

    lineLength = 0

    if indexTipPos != None and thumbTipPos != None:
        cx, cy = int((thumbTipPos[0] + indexTipPos[0]) // 2), int((thumbTipPos[1] + indexTipPos[1]) // 2)
        cv2.line(img, thumbTipPos, indexTipPos, (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        a = thumbTipPos[0] - indexTipPos[0] if thumbTipPos[0] > indexTipPos[0] else indexTipPos[0] - thumbTipPos[0]
        b = thumbTipPos[1] - indexTipPos[1] if thumbTipPos[1] > indexTipPos[1] else indexTipPos[1] - thumbTipPos[1]

        lineLength = int(math.sqrt(a * a + b * b)) - 20
        
        if lineLength > 100:
            volume = 100
        elif lineLength < 0:
            volume = 0
        else:
            volume = lineLength
        
        print(volume)

    # fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)