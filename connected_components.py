from __future__ import print_function
import cv2
import argparse
import numpy as np


frame = cv2.imread('sample_hand.jpg')

window_name = 'hand_finder'
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'

scale_percent = 20
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100) 
res_frame = cv2.resize(frame, (width, height))

def nothing(x):
    pass

cv2.namedWindow(window_name)
cv2.namedWindow('new_window')

cv2.createTrackbar('gauss_blur', window_name , 1, 20, nothing)  
cv2.createTrackbar('max_binary', window_name, 0, 255, nothing)

while True:
   

    gray = cv2.cvtColor(res_frame,cv2.COLOR_BGR2GRAY)  

    max_binary_value = cv2.getTrackbarPos('max_binary', window_name)

    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  

    try:
        ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
        markers = np.array(markers, dtype=np.uint8)  
        label_hue = np.uint8(179*markers/np.max(markers))  
        blank_ch = 255*np.ones_like(label_hue)  
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0  

        
        statsSortedByArea = stats[np.argsort(stats[:, 4])]  
        roi = statsSortedByArea[-3][0:4]  
        x, y, w, h = roi  
        subImg = labeled_img[y:y+h, x:x+w]  

        _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        ellipseParam = cv2.fitEllipse(contours[0])  
        subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
        subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
        cv2.imshow('new_window', subImg)
    except:
        cv2.imshow('new_window', thresh)
        pass

    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        break



