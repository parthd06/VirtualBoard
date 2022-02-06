import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

#######
brushThickness = 15
eraserThickness = 100
#######

# for getting the img from header folder
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (0, 0, 0)



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8 )


while True:

    #1. Import the Image
    success, img = cap.read()
    img = cv2.flip(img, 1)


    #2. Find Hand LandMarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        #tip of the index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which Fingers are Up (1 Fing to Draw 2 to Select)
        fingers = detector.fingersUp()
        # print(fingers)


        #4. If selection Mode - then select don't draw (ie. 2 Fingers are up only select)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            #Checking for the click on which marker our hand points
            if y1 < 125:
                if 250 < x1 < 400:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
                elif 500 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 700 < x1 < 800:
                    header = overlayList[2]
                    drawColor = (0, 69, 255)
                elif 900 < x1 < 1000:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 1025 < x1 < 1200:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        #5. If Drawing Mode - then only draw (ie. 1 Finger is up it should be used to draw)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (128, 128, 128):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)



    # Setting the Header Image (Default one)
    img[0:125, 0:1280] = header

    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)



