import numpy as np
import cv2
import sys
from random import randint
import csv

fp = open('dados/data/Results.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

TEXT_COLOR = (randint(0, 255), randint(0,255), randint(0,255))
BORDER_COLOR = (randint(0, 255), randint(0,255), randint(0,255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1.2
VIDEO = 'dados\Ponte.mp4'
TITLE_TEXT_POSITION = (100, 40)

algorithm_types = ['KNN','GMG', 'CNT', 'MOG', 'MOG2',]

def Subtrator(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Detector inválido')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO)

background_subtractor = []

for i, a in enumerate(algorithm_types):
    background_subtractor.append(Subtrator(a))

def main():
    frame_number = 0
    while (cap.isOpened):
        
        ok, frame = cap.read()

        if not ok:
            print('Erro na captura')
            break

        frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

        gmg = background_subtractor[0].apply(frame)
        mog = background_subtractor[1].apply(frame)
        mog2 = background_subtractor[2].apply(frame)
        knn = background_subtractor[3].apply(frame)
        cnt = background_subtractor[4].apply(frame)

    
        gmgCount = np.count_nonzero(gmg)
        mogCount = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)
        knnCount = np.count_nonzero(knn)
        cntCount = np.count_nonzero(cnt)

        writer.writerow({'Frame': 'GMG', 'Pixel Count': gmgCount})
        writer.writerow({'Frame': 'MOG', 'Pixel Count': mogCount})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2Count})
        writer.writerow({'Frame': 'KNN', 'Pixel Count': knnCount})
        writer.writerow({'Frame': 'CNT', 'Pixel Count': cntCount})

        cv2.putText(gmg, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(mog, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(mog2, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(knn, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(cnt, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Original', frame)
        cv2.imshow('GMG', gmg)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)


        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

main()  