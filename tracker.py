import cv2
import landmarks as lm
import time
cap = cv2.VideoCapture(0)

detector = lm.handDetector(maxHands=1)


while True:
    _,img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        print(x1, y1)
        cv2.circle(img,(x1,y1),color=(255,0,0),radius=10,thickness=5)


    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
