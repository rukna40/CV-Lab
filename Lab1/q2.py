import cv2 as cv

cap = cv.VideoCapture('/Lab1/assets/mixkit-52025-video-52025-hd-ready.mp4')

while True:
    _, frame = cap.read()
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# cv.destroyAllWindows()
cap.release()