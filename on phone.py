import cv2
import numpy as np
import os
import imageio
import imutils
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
import urllib.request

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
# def implier()
def nothing(x):
    pass
def empty(a) :
    pass
def getContours(img):
   _,contours,_ = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   for cnt in contours:
        area = cv2.contourArea(cnt)
#        print(area)
        if area>4:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#            peri = cv2.arcLength(cnt,True)v2.imwrite('Contours'+str(i)+'.jpg' , effect)
def funcBrightContrast(bright=0):
    bright = cv2.getTrackbarPos('bright', 'TrackBars')
    contrast = cv2.getTrackbarPos('contrast', 'TrackBars')

    effect = apply_brightness_contrast(img,bright,contrast)
    # ret,thresh1 = cv2.threshold(effect,0,255,cv2.THRESH_BINARY)
    # effect2 = apply_brightness_contrast(img2 , bright,contrast)
    # All_Frames_in_list.append(effect)
    # All_dot_likes_list.append(effect2)
    cv2.imshow('Effect', effect)

def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    # cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def powerlaw_gamma(imagef , gamma) :
    imagef = np.clip(np.power(imagef, gamma), 0, 255).astype('uint8')
    return(imagef)
gamma = 100
# def combiner(n,List1):
#     Gray_frame = cv2.cvtColor(List1[n-1], cv2.COLOR_BGR2GRAY)
#     ret,mask_combiner = cv2.threshold(Gray_frame,10,255,cv2.THRESH_BINARY)
#     mask_inv2 = cv2.bitwise_not(mask_combiner)
#     if n==0 :
#         return(cv2.bitwise_or(List1[n-1] , List1[n] , mask = mask_inv2))
#     else :
#         return(cv2.bitwise_or(combiner(n-1 , List1) , List1[n-2] , mask = mask_inv2))  
All_Frames_in_list = []
All_dot_likes_list = []
Allmasks = []
All=[]
New = []
New2 = []
New3 = []
New4 = []
New5 = []
New6 = []
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",   640  , 480)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
cv2.createTrackbar('R-low','TrackBars',0,255,empty)
cv2.createTrackbar('R-high','TrackBars',255,255,empty)
cv2.createTrackbar('G-low','TrackBars',0,255,empty)
cv2.createTrackbar('G-high','TrackBars',255,255,empty)  
cv2.createTrackbar('B-low','TrackBars',0,255,empty)
cv2.createTrackbar('B-high','TrackBars',255,255,empty)
cv2.createTrackbar("gamma, (* 0.01)", 'TrackBars', gamma, 200, nothing)
# cap = cv2.VideoCapture(2)
i = -1
k = -1
j = -1
# dim = (640,480)
# cap.set(cv2.CAP_PROP_EXPOSURE,-20)
# cap.set(cv2.CAP_PROP_BRIGHTNESS,16)
# cap.set(cv2.CAP_PROP_CONTRAST,32)
# cap.set(cv2.CAP_PROP_TEMPERATURE,6500)
# # cap.set(cv2.CAP_PROP_AUTOFOCUS,15)
# cap.set(cv2.CAP_PROP_SATURATION,15)
# cap.set(cv2.CAP_PROP_SHARPNESS,15)





url='http://21.46.176.154:8080/shot.jpg'

while(True):
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    dim = (640,480)
    resized = cv2.resize(img,dim)
    # put the image on screen
    # cv2.imshow('IPWebcam',resized)
    frame = resized
    # cap.set(cv2.CAP_PROP_EXPOSURE,2)
    # All.append(frame)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")
    rl = cv2.getTrackbarPos('R-low','TrackBars')
    rh = cv2.getTrackbarPos('R-high','TrackBars')
    gl = cv2.getTrackbarPos('G-low','TrackBars')
    gh = cv2.getTrackbarPos('G-high','TrackBars')
    bl = cv2.getTrackbarPos('B-low','TrackBars')
    bh = cv2.getTrackbarPos('B-high','TrackBars')
    gamma = cv2.getTrackbarPos("gamma, (* 0.01)",'TrackBars') * 0.01
    lower = np.array([h_min , s_min , v_min])
    upper = np.array([h_max , s_max , v_max])
    lower1 = np.array([rl,gl,bl])   
    upper1 = np.array([rh,gh,bh])
    framen = frame.copy()
    framen[:] = [bl,gl,rl]
    frame1 = cv2.filter2D(frame , -1 , kernel)
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    framenHSV = cv2.cvtColor(framen,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHSV, lower,upper)
    mask1 = cv2.inRange(framenHSV , lower1 , upper1)
    rgbimg = cv2.bitwise_and(framen, framen , mask = mask1)
    rgbimg1 = cv2.bitwise_and(frame ,framen , mask = mask1 )
    imgresult = cv2.bitwise_and( frame, frame , mask = mask)
    new_imgresult = cv2.bitwise_or(imgresult , frame , mask=None)
    imgresult2 = cv2.bitwise_and(frameHSV, frameHSV , mask = mask)
    rgbimg2 = cv2.bitwise_and(rgbimg , imgresult , mask = mask1)
    imgGray = cv2.cvtColor(imgresult2 , cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray , (7,7) , 1)
    imgCanny = cv2.Canny(imgBlur , 50 , 50)
    imgSharp = cv2.filter2D(imgresult , -1  , kernel)
    imgSharp2 = cv2.filter2D(imgresult2 , -1  , kernel)
    imgSharp3 = cv2.filter2D(imgCanny , -1 , kernel)
    imgContour = imgresult.copy()
    getContours(imgSharp3)
    imgCountour_SHARP = cv2.filter2D(imgContour , -1 , kernel)
    imgBlank = np.zeros_like(imgresult2)
    imgstack_new = stackImages(0.7 , ([imgresult , new_imgresult , rgbimg2],[imgContour , mask , imgresult2]))
    # gamma_imgresult = imgstack_new.copy()
    gamma_imgresult = imgstack_new.copy()
    Final_Sharpen = cv2.filter2D(gamma_imgresult , -1 , kernel)
    gamma_new_result = new_imgresult.copy()
    gamma_imgresult = powerlaw_gamma(gamma_imgresult , gamma)
    gamma_new_result = powerlaw_gamma(gamma_new_result, gamma)
    if __name__ == '__main__':
        original1 = gamma_imgresult
        img = original1.copy()
    cv2.namedWindow('TrackBars',1)
    bright = 254
    contrast = 127
    cv2.createTrackbar('bright', 'TrackBars', bright, 2*255, funcBrightContrast)
    cv2.createTrackbar('contrast', 'TrackBars', contrast, 2*127, funcBrightContrast)
    funcBrightContrast(0)
    k = cv2.waitKey(1) & 0xFF
    # if k != ord('q'):
        # i += 1
        # New6.append(combiner(i,All_Frames_in_list))
        # cv2.imwrite('bitwise'+str(i)+'.jpg' , New6[i])
    if k == ord("q"):
        break
# cap.release()
cv2.destroyAllWindows()
