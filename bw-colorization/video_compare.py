import cv2

# video source: https://www.tukuppt.com/videomuban/mp4/__zonghe_0_0_0_0_0_0_7.html
v1 = cv2.VideoCapture("videos/3.mp4")
v2 = cv2.VideoCapture("videos/3->cartoon.avi")

c = 0

while True:
    ret1, frame1 = v1.read()
    ret2, frame2 = v2.read()

    if ret1 and ret2:
        if c % 4 == 0:
            cv2.imshow("original", frame1)
            cv2.imshow("cartoon", frame2)
            cv2.waitKey(1) & 0xFF
        c = c + 1
    else:
        break
v1.release()
v2.release()
cv2.destroyAllWindows()
