import cv2
import cvzone.HandTrackingModule as HandTrackingModule


# Webcamm
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandTrackingModule.HandDetector(maxHands = 2, detectionCon = 0.8)

while True:
    # Get the frame from the weebcam
    success, img = cap.read()
    # Hands
    hands, img = detector.findHands(img)

    # Land

    cv2.imshow("Image", img)
    cv2.waitKey(1)


