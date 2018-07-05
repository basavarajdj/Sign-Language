import cv2
import numpy as np


cap = cv2.VideoCapture(0)
#fg = cv2.createBackgroundSubtractorMOG2()


x, y, w, h = 280, 50, 350, 350
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
imgNum = 0
frames = 0
training_size = 2000
start = False
while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.GaussianBlur(mask, (5,5),0)
    skinmask = cv2.erode(mask, kernel,iterations=2)
    skinmask = cv2.dilate(mask, kernel,iterations=2)
    #skinmask = cv2.morphologyEx(skinmask, cv2.MORPH_OPEN, kernel)
    
    AndRes = cv2.bitwise_and(frame, frame, mask=skinmask)
    gray = cv2.cvtColor(AndRes, cv2.COLOR_BGR2GRAY)
    
    _,thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,2)

    thresh = thresh[y:y+h, x:x+w]
    graySquare = gray[y:y+h, x:x+w]

    #img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img, contours, hierarchy = cv2.findContours(graySquare, 1,2)
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'ImgNum: '+str(imgNum),(30,400), font, 1,(0,255,0),2,cv2.LINE_AA)

    if (len(contours) > 0):
        contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(contour) > 10000 and frames > 100:
        cArea = cv2.contourArea(contour)

        x1, y1, w1, h1 = cv2.boundingRect(contour)
        imgNum +=1

        img2Save = graySquare[y1:y1+h1, x1:x1+w1]
        img2Save = cv2.resize(img2Save, (64, 64))

        cv2.imwrite("training_data/"+str(imgNum)+".png",img2Save)
    
    #fgmask = fg.apply(gray)

    cv2.imshow("frame",frame)
    cv2.imshow("graySquare", graySquare)

    if imgNum == training_size:
        break

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    if k == ord('s'):
        start = True

    if start:
        frames += 1
    
cap.release()
cv2.destroyAllWindows()
