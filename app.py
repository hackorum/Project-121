import cv2  
import numpy as np
import time

cap = cv2.VideoCapture(1)
image = cv2.imread('bg.jpg')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file = cv2.VideoWriter('invisible.avi', fourcc, 20.0, (640, 480))

time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))
    image = cv2.resize(image, (640, 480))
    frame = np.flip(frame, axis=1)
    # l_black = np.array([30, 30, 0])
    # u_black = np.array([104, 153, 70])
    l_black = np.array([0, 120, 70])
    u_black = np.array([180, 255, 255])
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), l_black, u_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame - res
    f = np.where(f == 0, image, f)
    # cv2.imshow("Real Video", frame)
    cv2.imshow("Masked Video", f)
    final_output = cv2.addWeighted(res, 1, res, 1, 0)
    out_file.write(final_output)
    k = cv2.waitKey(1)
    if k == 27 or k == 81:
        break
cap.release()
out_file.release()
cv2.destroyAllWindows()