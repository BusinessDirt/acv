import cv2
import mediapipe as mp
import time

class FaceDetector():

    def __init__(self, detectionConfidence = 0.5):
        self.detectionConfidence = detectionConfidence

        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(self.detectionConfidence)

    def drawBoundsToImage(self, img, fancy = False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)


        if self.results.detections:
            for boundingBox in enumerate(self.getFaceBounds(img)):
                if fancy:
                    self.drawFancyBoundingBox(img, boundingBox[1][1])
                else:
                    cv2.rectangle(img, boundingBox[1][1], (255,0, 255), 1)
                    
                cv2.putText(img, f'{int(boundingBox[1][2][0] * 100)}%', (boundingBox[1][1][0], boundingBox[1][1][1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img

    def drawFancyBoundingBox(self, img, boundingBox, l = 30, t = 5, rt = 1):
        x, y, w, h = boundingBox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, boundingBox, (255,0, 255), rt)

        # top left
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)


    def getFaceBounds(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)

        boundingBoxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                tmpBoundingBox = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                boundingBox = int(tmpBoundingBox.xmin * iw), int(tmpBoundingBox.ymin * ih), \
                    int(tmpBoundingBox.width * iw), int(tmpBoundingBox.height * ih)
                boundingBoxes.append([id, boundingBox, detection.score])

        return boundingBoxes


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img = detector.drawBoundsToImage(img)
        # print(detector.getFaceBounds(img))

        # fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()