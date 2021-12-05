import cv2
import mediapipe as mp
import time

from enum import Enum

from mediapipe.framework.formats.landmark_pb2 import LandmarkList

class PoseLandmark(Enum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class PoseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionConfidence, self.trackingConfidence)

    def drawPoseToImage(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPoseLandmarks(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                
        return lmList

    def getLandmarkPosition(self, img, landmarkName = PoseLandmark.NOSE):
        landmarkList = self.getPoseLandmarks(img)
        if len(landmarkList) != 0:
            x, y = landmarkList[landmarkName.value][1], landmarkList[landmarkName.value][2]
            return [x, y]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])

        # fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()