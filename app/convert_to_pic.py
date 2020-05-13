import time
import numpy as np
import cv2

cap = cv2.VideoCapture('../data/video/video-new.mp4')


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    # Display the resulting frame
    if ret == True:
        print(str(time.time())+'.jpg')
        cv2.imwrite('../data/image/GH040022/'+str(time.time())+'.jpg', frame)
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
