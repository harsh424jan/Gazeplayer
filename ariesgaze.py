import numpy as np
import cv2
import os
import PIL
from PIL import Image
vidcap = cv2.VideoCapture(0)
import pyautogui
from keras.models import load_model
model = load_model('my_model1.h5')


# 1 -- top left    
# 2 -- top right    
# 3 -- bottom left
# 4 -- bottom right
cnt=0
pre =1
now=1
list =[]
list.append("top left")
list.append("top right")
list.append("bottom left")
list.append("bottom right")
#vidcap = cv2.VideoCapture('/home/aniruddha/Videos/Webcam/test.webm')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
success,image = vidcap.read()
idx=0
count = 0
success = True
while success:
  idx+=1
  success,image = vidcap.read()
  if idx%13==0:
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
    #print(image2.shape)
    vis = np.concatenate((image0, image1), axis=0)
    vis = np.concatenate((vis, image), axis=1)
    dim=(100,100)
    vis = cv2.resize(vis, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("image2.jpg",vis)
    im = PIL.Image.open('/home/aniruddha/image2.jpg')
    img = PIL.Image.open('/home/aniruddha/image2.jpg').convert("L")
    arr = np.array(img)
    arr = arr.reshape(1, 1, 100, 100).astype('float32')
    arr=arr/300
    now=np.argmax(model.predict(arr))
    print(now)
    if now==pre:
        cnt+=1
        if cnt>=2:
            print(now)
            if now==1:
                pyautogui.moveTo(400, 250)
            if now==2:
                pyautogui.moveTo(1400, 250)
            if now==3:
                pyautogui.moveTo(400, 800)
            if now==4:
                pyautogui.moveTo(1400, 800)
    else:
        cnt=0
        pre=now
    



# When everything done, release the capture

  #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file

