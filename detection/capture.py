import cv2 as cv

# 读取设备
cap = cv.VideoCapture(0)

fps = cap.get(cv.CAP_PROP_FPS)
print(fps)
# set dimensions 设置分辨率
cap.set(3, 1280)
cap.set(4, 720)

while True:
    # take frame 读取帧
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if ret:
        # write frame to file
        cv.imshow('result', frame)  # 截图
    if cv.waitKey(1) & 0xff == ord('q'):
        break
# release camera 必须要释放摄像头
cap.release()
cv.destroyAllWindows()
