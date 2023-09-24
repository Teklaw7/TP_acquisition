import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
cap = cv2.VideoCapture(4)
while(True):
    ret, frame = cap.read() #1 frame acquise par itération
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    # cap.set(cv2.CAP_PROP_FPS,60)
    # cap.set(cv2.CAP_PROP_SHARPNESS, 3)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS,100) # Gère l'exposition du flux
    # cap.set(cv2.CAP_PROP_SATURATION,100) # Gère la saturation du flux
    # cap.set(cv2.CAP_PROP_CONTRAST,100)
    key = cv2.waitKey(1)
    if key & 0xFF ==ord('q'):
        break
    framecvt =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key & 0xFF ==ord('s'):
        path = '/home/timothee.teyssier/Documents/home/Documents/TP_acquisition/picture'
        count = len(os.listdir(path))
        plt.imsave(path+f'/pic_{count}.png', framecvt)

    hist = cv2.calcHist([framecvt], [0],None, [256],[0,256])
    plt.plot(np.linspace(0,256,256), hist)
    plt.title('Histogramme')
    plt.draw()
    plt.pause(0.0001)
    plt.cla()
    gray = cv2.cvtColor(framecvt, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50,300, apertureSize=3)
    lines = cv2.HoughLinesP(edges,1, np.pi/180, 100, 100, 50)
    # print(lines)
    if not(lines is None):
        for line in lines:
            x1,y1, x2,y2 =line[0]
            # print(x1, x2, y1, y2)
            # res =cv2.line(frame, (x1,y1), (x2,y2), (0,255,0),2)
    orb = cv2.ORB_create(nfeatures=500, nlevels=8)
    kp = orb.detect(frame, None)
    # print(kp)
    res = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow('Capture_Video', res)

    height, width = frame.shape[:2]
    center = (width/2, height/2)
    tx, ty = 10, 15
    translation_matrix = np.array([[1,0,tx],[0,1,ty]], dtype=np.float32)
    theta = 180
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    rotation_matrix = cv2.getRotationMatrix2D(center= center, angle=180, scale=1)
    dst = cv2.warpAffine(frame,M= rotation_matrix, dsize=(width, height))
    cv2.imshow('Capture video translatee', dst)



cap.release()
cv2.destroyAllWindows()
