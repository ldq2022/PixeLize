import numpy as np
import cv2

# video source: https://www.tukuppt.com/videomuban/mp4/__zonghe_0_0_0_0_0_0_7.html
cap = cv2.VideoCapture("videos/2.mp4")

# property_id: 3 = CV_CAP_PROP_FRAME_WIDTH
# property_id: 4 = CV_CAP_PROP_FRAME_HEIGHT
# property_id: 5 = CV_CAP_PROP_FPS
# (see: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=capture)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create VideoWriter object

out = cv2.VideoWriter('videos/cartoon.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if ret:
        # this is the algo from portrait.py (see portrait.py for more details)
        numDownSamples = 2
        numBilateralFilters = 50

        # Step 1: downsample (reduce color variance)
        img_color = frame
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)


        # Step 2: bilateral filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)


        # Step 3: upsample
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)


        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # need to revert to original colorformat in order to properly write
        # otherwise out.write() won't work
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        cartoon = cv2.bitwise_and(img_edge, img_color)
        cartoon = cv2.resize(cartoon, (frame_width, frame_height))
        cv2.imshow("original", frame)
        cv2.imshow("cartoon", cartoon)
        cv2.waitKey(1) & 0xFF
        out.write(cartoon)


    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print('process finished')
