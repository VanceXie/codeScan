import cv2

capture=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')#定义编码输出器
out = cv2.VideoWriter(r'/home/ubei/Pictures/VideoTrait.avi',fourcc, 20.0, (640,480))#定义输出文件名称及帧大小
while capture.isOpened:
    ret,frame=capture.read()
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGRA2YUV)#转化为YUV编码
        gray=cv2.flip(gray,1)#水平翻转
        out.write(gray)#将每帧图像写出
        cv2.imshow('frame',gray)#显示图像
        if cv2.waitKey(1)&0xFF==ord('q'):#按‘q’键结束
            break
    else:
        break
capture.release()
out.release()
cv2.destroyAllWindows()