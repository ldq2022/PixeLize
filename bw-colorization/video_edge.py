import numpy as np
import cv2

# video source: https://www.tukuppt.com/videomuban/mp4/__zonghe_0_0_0_0_0_0_7.html
cap = cv2.VideoCapture("videos/3.mp4")

# property_id: 3 = CV_CAP_PROP_FRAME_WIDTH
# property_id: 4 = CV_CAP_PROP_FRAME_HEIGHT
# property_id: 5 = CV_CAP_PROP_FPS
# (see: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=capture)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create VideoWriter object

out = cv2.VideoWriter('videos/edge.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                      (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        img_edge = cv2.resize(img_edge, (frame_width, frame_height))
        # need to revert to original colorformat in order to properly write
        # otherwise out.write(colorized) won't work
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        cv2.imshow("original", frame)
        cv2.imshow("edge", img_edge)
        cv2.waitKey(1) & 0xFF
        out.write(img_edge)


    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print('process finished')
