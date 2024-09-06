import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/Fist"
counter = 0

while True:
    success, img = cap.read()
    flip_img = cv2.flip(img, 1)
    hands, img = detector.findHands(flip_img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = flip_img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            constant = imgSize / h
            wCalc = math.ceil(constant * w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCalc)/2)
            imgWhite[:, wGap:wCalc+wGap] = imgResize

        else:
            constant = imgSize / w
            hCalc = math.ceil(constant * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCalc)/2)
            imgWhite[hGap:hCalc+hGap, :] = imgResize

        cv2.imshow("image white", imgWhite)

    cv2.imshow("Hand Gesture Project", flip_img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_Sample{time.time()}.jpg', imgWhite)
        print(counter)