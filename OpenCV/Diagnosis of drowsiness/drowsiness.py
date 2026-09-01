import os
import cv2 as cv
import numpy as np
import time
from pygame import mixer
import tensorflow as tf
from tensorflow.keras.models import load_model




mixer.init()
sound=mixer.Sound('alarm.wav')

casecade_pass=cv.data.haarcascades
face_cascade = cv.CascadeClassifier(os.path.join(casecade_pass ,"haarcascade_frontalface_default.xml"))
l_eye_cascade = cv.CascadeClassifier(os.path.join(casecade_pass ,'haarcascade_lefteye_2splits.xml'))
r_eye_cascade = cv.CascadeClassifier(os.path.join(casecade_pass ,'haarcascade_righteye_2splits.xml'))

label=['close' , 'open']
model=load_model('models/cnnCat2.keras')
path=os.getcwd()

video_cap=cv.VideoCapture(0)
font=cv.FONT_HERSHEY_COMPLEX_SMALL

count=0
closed_duration = 0
closed_start_time = None

thicc=2
rpred=[99]
lpred=[99]


while(True):
    ret , frame=video_cap.read()
    h , w=frame.shape[:2]
    
    gray=cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray , minNeighbors=5 , scaleFactor=1.1 , minSize=(25,25))
    left_eye=l_eye_cascade.detectMultiScale(gray)
    right_eye=r_eye_cascade.detectMultiScale(gray)
    
    
    cv.rectangle(frame , (0,h-50),(200,h) ,(0,0,0) , thickness=cv.FILLED)
    
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y) , (x+w,y+h) , (220,220,10),1)
    
    for (x,y,w,h) in right_eye:
        cv.rectangle(frame,(x,y) , (x+w,y+h) , (220,220,10),1)
        r_eye=frame[y:y+h,x:x+w]
        count+=1
        r_eye=cv.cvtColor(r_eye,cv.COLOR_BGR2GRAY)
        r_eye=cv.resize(r_eye,(24,24))
        r_eye=r_eye/255
        r_eye=r_eye.reshape(24,24,-1)
        r_eye=np.expand_dims(r_eye,axis=0)
        rpred=np.argmax(model.predict(r_eye),axis=1)
        print(rpred)
        if (rpred[0]==1):
            label='open'
        
        if (rpred[0]==0):
            label='closed'
        break
        
        
    for (x,y,w,h) in left_eye:
        cv.rectangle(frame,(x,y) , (x+w,y+h) , (220,220,10),1)
        l_eye=frame[y:y+h,x:x+w]
        count+=1
        l_eye=cv.cvtColor(l_eye,cv.COLOR_BGR2GRAY)
        l_eye=cv.resize(l_eye,(24,24))
        l_eye=l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye=np.expand_dims(l_eye,axis=0)
        lpred=np.argmax(model.predict(l_eye),axis=1)
        print(lpred)
        
        if (lpred[0]==1):
            label='open'
        
        if (lpred[0]==0):
            label='closed'
        break
    
    if (rpred[0]==0 and lpred[0]==0):
        
        if closed_start_time is None:
            closed_start_time = time.time()

        closed_duration = time.time() - closed_start_time
        
        cv.putText(frame,'closed',(50,h-20) , font, 1 , (0,0,255),1,cv.LINE_AA)
        
    else:
        closed_start_time = None
        cv.putText(frame,'open',(50,h-20) , font, 1 , (0,255,0),1,cv.LINE_AA)
        
        
    
    cv.putText(frame,'score'+str(closed_duration),(150,h-20) , font, 1 , (255,255,255),1,cv.LINE_AA)
        
    if closed_duration >= 5:
        cv.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
        except:
            pass
        
        if(thicc<10):
            thicc+=2
        else:
            thicc-=2
            if(thicc<2):
                thicc=2
        h , w=frame.shape[:2]       
        cv.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),thicc)
        
    cv.imshow('frame',frame)
    keyexit = cv.waitKey(5) & 0xFF

    # Exit the loop when the ESC key (ASCII code 27) is pressed
    if keyexit == 27:
        break
    
video_cap.release()
cv.destroyAllWindows()
                
                