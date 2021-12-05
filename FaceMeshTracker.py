import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, staticMode = False, maxFaces = 2, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.detectionConfidence, self.trackingConfidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    def getFaceMeshes(self, img):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLandmark in self.results.multi_face_landmarks:

                face = []
                for id, landMark in enumerate(faceLandmark.landmark):
                    h, w, c = img.shape
                    x, y = int(landMark.x * w), int(landMark.y * h)
                    face.append([id, x, y])

                faces.append([faceLandmark, face])

        return faces
    
    def drawFaceMeshesToImage(self, img):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        if self.results.multi_face_landmarks:
            for faceLandmark in self.getFaceMeshes(img):
                self.mpDraw.draw_landmarks(img, faceLandmark[0], self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

        return img

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = detector.drawFaceMeshesToImage(img)

        # fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()