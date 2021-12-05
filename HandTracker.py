import cv2
import mediapipe as mp
import time

from enum import Enum

class HandLandmark(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class HandDetector():

    def __init__(self, staticMode = False, maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.staticMode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def drawLandmarksToImage(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def getHandLandmarks(self, img, handNumber = 0):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(hand.landmark):
                h, w, c =img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList

    def getLandmarkPosition(self, img, handNumber = 0, landmarkName = HandLandmark.WRIST):
        landmarkList = self.getHandLandmarks(img, handNumber)
        if len(landmarkList) != 0:
            x, y = landmarkList[landmarkName.value][1], landmarkList[landmarkName.value][2]
            return [x, y]



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.drawLandmarksToImage(img)
        lmList = detector.getHandLandmarks(img)
        
        print(detector.getLandmarkPosition(img, 0, HandLandmark.THUMB_TIP))

        # fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()