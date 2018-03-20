import cv2
import numpy as np
import math
import time
import os
import pyautogui


from keras.models import load_model
model = load_model('my_model1.h5')


import PIL
time.sleep(1)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
success,image = cap.read()

list =[]
list.append("top left")
list.append("top right")
list.append("bottom left")
list.append("bottom right")
cnt=0
pre =1
now=1

ret1, img1 = cap.read()
idx=0
count = 0
success = True

# cv2.rectangle(img1, (250,300), (0,0), (0,255,0),0)
# crop_img1 = img1[0:300, 0:250]
# grey1 = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
# value = (35, 35)
# blurred1 = cv2.GaussianBlur(grey1, value, 0)
# _, thresh2 = cv2.threshold(blurred1, 127, 255,
# cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

def get_quad() :
    #idx+=1
    success,image = cap.read()
    #if idx%13==0:
    cv2.imwrite("image5.jpg", image)
    img = cv2.imread('image5.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        
        dim=(140,200)
        resized = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("image.jpg",resized)
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        count1=0
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_eyes=roi_gray[ey:ey+eh, ex:ex+ew]
            dim=(70,100)
            resized = cv2.resize(roi_eyes, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite("image%d.jpg"%count1,resized)
            count1+=1
    image0=cv2.imread("image0.jpg")
    image1=cv2.imread("image1.jpg")
    image=cv2.imread("image.jpg")
    #print(image0.shape)
    #print(image1.shape)
    #print(image.shape)
    vis = np.concatenate((image0, image1), axis=0)
    vis = np.concatenate((vis, image), axis=1)
    dim=(100,100)
    vis = cv2.resize(vis, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("image2.jpg",vis)
    im = PIL.Image.open('image2.jpg')
    img = PIL.Image.open('image2.jpg').convert("L")
    arr = np.array(img)
    arr = arr.reshape(1, 1, 100, 100).astype('float32')
    arr=arr/300
    now=np.argmax(model.predict(arr))
    return now


def get_fingers ():

    ret, img = cap.read()
    cv2.rectangle(img, (320,300), (0,0), (0,255,0),0)
    crop_img = img[0:300, 0:320]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #grey = grey - grey1
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)


    _, thresh1 = cv2.threshold(blurred, 127, 255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    

    
    #cv2.imshow('Thresholded', thresh1)
    im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


    cnt = max(contours, key = lambda x: cv2.contourArea(x))


    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)


    hull = cv2.convexHull(cnt)


    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)


    hull = cv2.convexHull(cnt, returnPoints=False)


    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

           
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            if a>1000:
                continue
            elif angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0,0,255], -1)
           
            cv2.line(crop_img,start, end, [0,255,0], 2)
           
        

       
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        #cv2.imshow('Contours', all_img)
        if count_defects  < 5:
            return count_defects+1
        else:
            return 0

    

  
cnt1=0
cnt2=0
cnt3=0
cnt4=0
count2=0

popup=0
play=0
idx=0

while (cap.isOpened()==True or success):
  idx+=1
  finger = get_fingers()
  now=get_quad()
  if idx%5==0:    
    count2+=1

    if(now==1):
        cnt1+=1
    elif(now==2):
        cnt2+=1
    elif(now==3):
        cnt1+=1
    elif(now==4):
        cnt2+=1
    now1=now

    if count2%1==0:
        now=max(cnt4,cnt3,cnt2,cnt1)
        #print(idx)
        print(list[now1-1])
        print(finger)
        
        

        if popup==0:
            if now==cnt1:
                pyautogui.moveTo(25, 476)
                #pyautogui.moveTo(90, 350)
            if now==cnt2:
                pyautogui.moveTo(1890, 476)
                #pyautogui.moveTo(1341, 354)
            if finger==2:
                if now==cnt1:
                    pyautogui.moveTo(25, 476)
                    #pyautogui.moveTo(90, 350)
                    #time.sleep(2)
                elif now==cnt2:
                    pyautogui.moveTo(1890, 476)
                    #pyautogui.moveTo(1341, 354)

                    #time.sleep(2)
                time.sleep(2)
            if finger==4:
                #pyautogui.click(700, 354)
                pyautogui.click(800, 505)
                popup=1
                time.sleep(3)
                continue


         
        elif popup!=0 :
            # if finger==2 and play==0:
            #     pyautogui.click(501, 430)
            #     #pyautogui.click(708,434)
            #     play=1
            #     time.sleep(2)
            # elif finger==2 and play==1:
            #     pyautogui.click(501, 430)
            #     #pyautogui.click(708,434)
            #     play=0
            #     time.sleep(2)

            if finger==4 and popup==1:
                popup=0
                pyautogui.click(1685,400)
                #pyautogui.click(1220,270)
                time.sleep(3)
                


    idx=0
    cnt1=0
    cnt4=0
    cnt3=0
    cnt2=0
    count2=0

    
    


  k = cv2.waitKey(10)
  if k == 27:
    break
  
    

