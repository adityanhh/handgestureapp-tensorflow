import cv2
import pywhatkit
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from datetime import datetime
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

count = 0
first = False
second = False
third = False

labels = ["OpenHand", "Fist", "CloseThumbs"]
currentTime = datetime.now()

while True:
    success, flip_img = cap.read()
    imgOutput = flip_img.copy()
    hands, flip_img = detector.findHands(flip_img)

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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            constant = imgSize / w
            hCalc = math.ceil(constant * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCalc)/2)
            imgWhite[hGap:hCalc+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if index == 0:
            while count < 1:
                first = True
                print('OpenHand')
                count += 1

        if index == 2:
            if first == True:
                while count < 2:
                    second = True
                    print('CloseThumbs')
                    count += 1

        if index == 1:
            if first == True and second == True:
                while count < 3:
                    third = True
                    print('Fist')
                    count += 1

        if first == True and second == True and third == True:
            while count < 4:
                print('Emergency')
                Msg = "[Automated Message] I'm in Danger!!!"
                pywhatkit.sendwhatmsg_instantly("+62 895612532800", Msg, 8, True)
                count += 1
                exit()

        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 2)

    cv2.imshow("Hand Gesture Project", imgOutput)
    cv2.waitKey(1)