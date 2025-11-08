import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os
from keras.models import load_model
import traceback


#model = load_model('cnn9.h5')

capture = cv2.VideoCapture(0)

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 30
step = 1
flag=False
suv=0
#Project directory
white=np.ones((400,400),np.uint8)*255
cv2.imwrite("white.jpg",white)


while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands= hd.findHands(frame, draw=False, flipType=True)

        if hands:
            # #print(" --------- lmlist=",hands[1])
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread("white.jpg")
            # img_final=img_final1=img_final2=0
            handz = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                # x1,y1,w1,h1=hand['bbox']

                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1),
                         (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
                         3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                cv2.imshow("Skeleton", white)
                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not os.path.exists(f"AtoZ/{chr(ord('A') + suv)}"):
                        os.makedirs(f"AtoZ/{chr(ord('A') + suv)}")
                    cv2.imwrite(f"AtoZ/{chr(ord('A') + suv)}/{step}.jpg", white)
                    print(f"Saved: AtoZ/{chr(ord('A') + suv)}/{step}.jpg")
                    step += 1
                elif key == ord('n'):
                    suv += 1
                    step = 1
                    print(f"Switched to letter: {chr(ord('A') + suv)}")
                    if suv >= 26:
                        print("All letters completed!")
                        break

    except Exception as e:
        print(f"Error: {e}")
        continue

capture.release()
cv2.destroyAllWindows()

