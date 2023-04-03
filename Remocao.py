import numpy as np
import cv2 
from time import sleep

VIDEO= "dados/Rua.mp4"
delay = 10

cap =  cv2.VideoCapture(VIDEO)
hasframe, frame = cap.read()


framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasframe, frame = cap.read()
    frames.append(frame)
    
medianFrame = np.median(frames,axis= 0).astype(dtype= np.uint8)
# print(medianFrame)
# cv2.imshow('Median Frame', medianFrame)
# cv2.waitKey(0)

cv2.imwrite('dados/median_frame.jpg', medianFrame)
'''Escala de Cinza'''


cap.set(cv2.CAP_PROP_POS_FRAMES,0)
grayMedianFrame= cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Median Frame Gray', grayMedianFrame)
# cv2.waitKey(0)
cv2.imwrite('dados/median_frame_gray.jpg', grayMedianFrame)


while (True):
    periodo = float(1 / delay)
    sleep(periodo)
    
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Acabou os frames')
        break
    '''Destacar os carros com transformações do cv2'''
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dframe = cv2.absdiff(frameGray, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('Frames em Cinza', dframe)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()