from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui
from collections import deque
import math

# frame = cv2.imread('5.jpg')
cam = cv2.VideoCapture(0)

window_name = 'hand_finder'

trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'

# scale_percent = 30
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100) 
# res_frame = cv2.resize(frame, (width, height))

paused = False
recentlyActed = False

def nothing(x):
    pass

q=0

lower_h_h, lower_h_s, lower_h_v = 96, 0, 35
upper_h_h, upper_h_s, upper_h_v = 155, 175, 248
lower_y_y, lower_y_r, lower_y_b = 0, 0, 125, 
upper_y_y, upper_y_r, upper_y_b = 255, 255, 167
max_binary_value = 250

cv2.namedWindow(window_name)
cv2.namedWindow('new_window')
cv2.createTrackbar('lower_h_h', window_name , lower_h_h, 255, nothing) 
cv2.createTrackbar('lower_h_s', window_name , lower_h_s, 255, nothing) 
cv2.createTrackbar('lower_h_v', window_name , lower_h_v, 255, nothing) 
cv2.createTrackbar('upper_h_h', window_name , upper_h_h, 255, nothing) 
cv2.createTrackbar('upper_h_s', window_name , upper_h_s, 255, nothing) 
cv2.createTrackbar('upper_h_v', window_name , upper_h_v, 255, nothing)  

cv2.createTrackbar('lower_y_y', window_name , lower_y_y, 255, nothing) 
cv2.createTrackbar('lower_y_r', window_name , lower_y_r, 255, nothing) 
cv2.createTrackbar('lower_y_b', window_name , lower_y_b, 255, nothing)
cv2.createTrackbar('upper_y_y', window_name , upper_y_y, 255, nothing) 
cv2.createTrackbar('upper_y_r', window_name , upper_y_r, 255, nothing) 
cv2.createTrackbar('upper_y_b', window_name , upper_y_b, 255, nothing)  

cv2.createTrackbar('gauss_blur', window_name , 1, 30, nothing)  
cv2.createTrackbar('max_binary', window_name, max_binary_value, 255, nothing)

gestureDeque = deque()
fingerCountDeque = deque()
handContourDeque = deque()
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB

prevAngle = None
prevPos = None
prevFingers = 0
avgFingerCount = 0


def rotated(angle, prevAngle, direction):
    if direction == 'left':
        return (angle - prevAngle > 10)
    else:
        return (angle - prevAngle < -10)

def moved(pos, prevPos, direction):
    if direction == 'left':
        return (pos[0] - prevPos[0] < -2)
    else:
        return (pos[0] - prevPos[0] > 2)

ms = 0

looping = False

while True:
    ms += 1

    ellipseDetected, handDetected = False, False
    ret, res_frame = cam.read()
    if not ret:
        continue

    lower_h_h = cv2.getTrackbarPos('lower_h_h', window_name)
    lower_h_s = cv2.getTrackbarPos('lower_h_s', window_name)
    lower_h_v = cv2.getTrackbarPos('lower_h_v', window_name)
    upper_h_h = cv2.getTrackbarPos('upper_h_h', window_name)
    upper_h_s = cv2.getTrackbarPos('upper_h_s', window_name)
    upper_h_v = cv2.getTrackbarPos('upper_h_v', window_name)

    lower_y_y = cv2.getTrackbarPos('lower_y_y', window_name)
    lower_y_r = cv2.getTrackbarPos('lower_y_r', window_name)
    lower_y_b = cv2.getTrackbarPos('lower_y_b', window_name)
    upper_y_y = cv2.getTrackbarPos('upper_y_y', window_name)
    upper_y_r = cv2.getTrackbarPos('upper_y_r', window_name)
    upper_y_b = cv2.getTrackbarPos('upper_y_b', window_name)

   

    lower_HSV = np.array([lower_h_h, lower_h_s, lower_h_v], dtype = "uint8")  
    upper_HSV = np.array([upper_h_h, upper_h_s, upper_h_v], dtype = "uint8")  
    
    convertedHSV = cv2.cvtColor(res_frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
    
    
    lower_YCrCb = np.array((lower_y_y, lower_y_r, lower_y_b), dtype = "uint8")  
    upper_YCrCb = np.array((upper_y_y, upper_y_r, upper_y_b), dtype = "uint8")  
        
    convertedYCrCb = cv2.cvtColor(res_frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
    
    skinMask = cv2.bitwise_and(skinMaskHSV,skinMaskYCrCb)  

    # skinMask = skinMaskYCrCb
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
    
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    blur_value = cv2.getTrackbarPos('gauss_blur', window_name)
    blur_value = blur_value+ (  blur_value%2==0)

    skinMask = cv2.GaussianBlur(skinMask, (blur_value, blur_value), 0) 
    skin = cv2.bitwise_and(res_frame, res_frame, mask = skinMask) 

    gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  

    max_binary_value = cv2.getTrackbarPos('max_binary', window_name)

    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  

    handFound = False
   
    """PART 2"""
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)
    
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  

    

    statsSortedByArea = stats[np.argsort(stats[:, 4])]  
        
    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            (ellipseX, ellipseY), (ellipseMA, ellipsema), ellipseAngle = ellipseParam
            if (ellipseMA/ellipsema < 8) and (ellipsema/ellipseMA < 8):
                cv2.imshow("ROI", subImg)  
                ellipseDetected, handDetected = True, True
            
            
        except:  
            pass
            
    


    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_OTSU )  
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    fingerCount = 0
    if len(contours)>=1:  
        largestContour = contours[0]          

        hull = cv2.convexHull(largestContour, returnPoints = False) 
        hullPoints = []

        M = cv2.moments(largestContour)  
        cX = int(M["m10"] / M["m00"])  
        cY = int(M["m01"] / M["m00"])

        handDetected = True

        for cnt in contours[:1]:
            defects = cv2.convexityDefects(cnt,hull) 
        
            if(not isinstance(defects,type(None))):  
                detected_fingers = []

                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])                      


                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  

                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))   
                                        
                    if (angle <= np.pi / 2.4):  
                        detected_fingers.append((a_squared, b_squared, far))

                    hullPoints.append(start)

                if len(detected_fingers) > 0:
                    # extra heuristic: make a second pass over, and filter out the fingers that have an abnormally long/short a/b
                    detected_fingers = sorted(detected_fingers, key=lambda finger: finger[1])
                    median_finger_length = detected_fingers[len(detected_fingers)//2][1]
                    for f in detected_fingers:
                        # if finger (Squared) is within like, 25% of the median finger (originalfinger length is within 50%) we're good
                        if f[1]/median_finger_length > 0.25:
                            fingerCount += 1  
                            # cv2.circle(thresh, f[2], 10, [255, 0, 0], -1)
                            
                    
                    fingerCount += 1 #just to add the "extra" finger
            

            # edge case of 1 finger - check the convex hull to see if there is a single acute angle that is very long
            # requires "clustering" close points on the hull together...
            hullPoints = np.float32(hullPoints)
            epsilon = 0.03*cv2.arcLength(hullPoints,False)
            hullPoints = cv2.approxPolyDP(hullPoints, epsilon, False)
            if fingerCount == 0:
                for j in range(len(hullPoints) - 2):
                    start, end, far = hullPoints[j][0], hullPoints[j+2][0], hullPoints[j+1][0]
                    start, end, far = tuple(start), tuple(end), tuple(far)

                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  

                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))  
                    if (angle <= np.pi / 2.4):
                        
                        fingerCount = 1
        

    if ellipseDetected:
        gestureDeque.append((ellipseX, ellipseY, ellipseMA, ellipsema, ellipseAngle))
    else:
        gestureDeque.append((0, 0, 0, 0, 0))
    
    if len(gestureDeque) < 30:
        pass
    else:
        if len(gestureDeque) > 30:
            gestureDeque.popleft()
        averageGestureVals = [sum([g[i] for g in gestureDeque])/len(gestureDeque) for i in range(5)]
        avgEllipseX, avgEllipseY, avgEllipseMA, avgEllipsema, avgEllipseAngle = averageGestureVals
        gestureString = "x: {}, y: {}".format(avgEllipseX, avgEllipseY)
        gestureString2 = "MA: {}, ma: {}, angle: {}".format(avgEllipseMA, avgEllipsema, avgEllipseAngle)
        cv2.putText(thresh, gestureString, org=(10,60), color=(255, 255, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
        cv2.putText(thresh, gestureString2, org=(10,80), color=(255, 255, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)



    if fingerCount <= 5:    
        fingerCountDeque.append(fingerCount) # fingercount will be zero if a hand isn't detected

    if len(fingerCountDeque) < 30:
        pass
    else:
        if len(fingerCountDeque) > 30:
            fingerCountDeque.popleft()
        # avgFingerCount = int(round(sum(fingerCountDeque)/len(fingerCountDeque)))
        avgFingerCount = max(set(fingerCountDeque), key=fingerCountDeque)
        cv2.putText(thresh, str(avgFingerCount)+ " fingers", org=(10,30), color=(255, 255, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)



    if handDetected:
        if len(handContourDeque) >= 2:
            lastCX, lastCY = handContourDeque[-1]
            if lastCX != 0 and abs(cX - lastCX)/lastCX < 0.1 and abs(cY - lastCY)/lastCY < 0.1:
                handContourDeque.append((cX, cY))
        else:
            handContourDeque.append((cX, cY))

    if len(handContourDeque) < 30:
        pass
    else:
        if len(handContourDeque) > 30:
            handContourDeque.popleft()
        avgCX, avgCY = [sum([c[i] for c in handContourDeque])/len(handContourDeque) for i in range(2)]
        contourString = "cX: {}, cY: {}".format(avgCX, avgCY)
        cv2.putText(thresh, contourString, org=(10,40), color=(255, 255, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
 

        


    # how do we check ? We check vs the average from like, half a second ago, and see if it is more than some threshold away
    # update the "previous" every half second
    # as soon as the hand or ellipse disappears (fingercount zero), then reset the "previous" to None
    
    if ms == 30:
            # do complex multi-frame ellipsedetection stuff
            # complex gesture - seek left/right with circle + hand movement left right
            # complex gesture - loop/unloop song (ellipse, 3->0 fingers)
            # complex gesture - shuffle playlist (go 3->0, no ellipse)

        if ellipseDetected:
            if (prevAngle and prevPos):
                if moved((avgEllipseX, avgEllipseY), prevPos, 'left'):
                    pyautogui.keyDown('shift')
                    pyautogui.press('right')
                    pyautogui.keyUp('shift')
                    print('seek left')
                elif moved((avgEllipseX, avgEllipseY), prevPos, 'right'):
                    pyautogui.keyDown('shift')
                    pyautogui.press('left')
                    pyautogui.keyUp('shift')    
                    print('seek right')   
                elif rotated(avgEllipseAngle, prevAngle, 'left'):
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('up')
                    pyautogui.keyUp('ctrl')
                    # print('rotateleft')
                elif rotated(avgEllipseAngle, prevAngle, 'right'):
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('down')
                    pyautogui.keyUp('ctrl')
                    # print('rotateright')         
                elif (prevFingers == 3) and (avgFingerCount == 0):
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('r')
                    if not looping:
                        pyautogui.press('r')
                        looping = True
                    pyautogui.keyUp('ctrl')

            prevAngle = avgEllipseAngle
            prevPos = (avgEllipseX, avgEllipseY)    
            prevFingers = avgFingerCount    
        else:
            prevAngle = None
            prevPos = None
            print('rest')
            # if (prevFingers == 3) and (avgFingerCount == 0):
            #     pyautogui.keyDown('ctrl')
            #     pyautogui.press('s')
            #     pyautogui.keyUp('ctrl')

            prevFingers = avgFingerCount


        if not recentlyActed:
            
            if (round(avgFingerCount) == 5):
                pyautogui.press('space')
                recentlyActed = True
                print('5')

        if (round(avgFingerCount) == 0) and not ellipseDetected:
            recentlyActed = False 

        ms = 0
    
    
        

    cv2.imshow("new_window", thresh)


    k = cv2.waitKey(1)
    if k == 27 or k == 113:
        cv2.destroyAllWindows()
        break

    



    
    




