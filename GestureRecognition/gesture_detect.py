import cv2
import numpy as np
import codecs, json
import time
import argparse
import os, shutil
def nothing(x):
    pass

def detect():
    cv2.namedWindow('Variable Values')
    cv2.createTrackbar('blur', 'Variable Values',11,179,nothing)
    cv2.createTrackbar('wait', 'Variable Values', 25, 100, nothing)
    cv2.createTrackbar('mask', 'Variable Values', 20, 255, nothing)
    cv2.createTrackbar('mask_lower', 'Variable Values',0, 255, nothing)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not os.path.exists("videos"):
        os.makedirs("videos")

    cycle = 1
    calibrated_x = calibrated_y = calibrated_h = calibrated_w = 0
    cap = cv2.VideoCapture(0)
    #Set frame size and initial values
    cap.set(3, 1280)
    cap.set(4, 1024)
    while(True):
        start_time = time.time()
        #Capture frames from the camera
        ret, frame = cap.read()
        hand = frame
        #Use Haar Cascade to detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #face_json = {'face_center_coordinate': []}
        #Draw the region of interest 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            center_face = (int(x+w/2),int(y+h/2))
            center_face_str = np.array(map(str, center_face))
        blur_val = cv2.getTrackbarPos('blur', 'Variable Values')

        #Blur the image to remove noised
        blur = cv2.blur(frame,(blur_val,blur_val))
        #blur = cv2.GaussianBlur(frame, (blur_val,blur_val), 0)

        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)\
        #Mask blurred image with Hue values of 0-20.
        mask_hsv = cv2.getTrackbarPos('mask','Variable Values')
        mask_lower = cv2.getTrackbarPos('mask_lower','Variable Values')
        mask = cv2.inRange(hsv,np.array([mask_lower,50,50]),np.array([mask_hsv,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Morph the image to get rid of noise.
        #Dilation increases the noise while erosion erases noise that was enlarged.
        dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1) 
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)

        #Find contours of the filtered frame
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Find largest contour on the frame
        max_area=100
        i=0 
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                i = n

        if len(contours) > 0:        
            cnts = contours[i]
        else:
            continue

        #Find convex hull of hand contour
        hull = cv2.convexHull(cnts)
        
        #Find convex defects of the hand
        hull2 = cv2.convexHull(cnts, returnPoints = False)
        defects = cv2.convexityDefects(cnts, hull2)
        
        #Get defect points and draw them in the frame
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            cv2.circle(frame,far,5,[150,255,255],2)

        #Find center of hand
        moments = cv2.moments(cnts)
        if moments['m00']!=0:
            x = int(moments['m10']/moments['m00'])
            y = int(moments['m01']/moments['m00'])
        center=(x,y)    

        #Draw center mass
        cv2.circle(frame,center,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,'Center',tuple(center),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            center = np.array(center)
            distance = np.sqrt(np.power(x[0]-center[0],2)+np.power(x[1]-center[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]

        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0,len(fingers)):
            #data.write("(" + str(fingers[i][0]) + ", " + str(fingers[i][1]) + ") ")
            distance = np.sqrt(np.power(fingers[i][0]-center[0],2)+np.power(fingers[i][1]-center[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0

        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1

        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        if cycle == 1:
            calibrated_x = x
            calibrated_y = y
            calibrated_h = h
            calibrated_w = w
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,255,255),2)
        
        cycle = cycle + 1
        #close the output video by pressing 'ESC'
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def detect(filename):
    """
    cv2.namedWindow('Variable Values')
    cv2.createTrackbar('blur', 'Variable Values',11,179,nothing)
    cv2.createTrackbar('wait', 'Variable Values', 25, 100, nothing)
    cv2.createTrackbar('mask', 'Variable Values', 20, 255, nothing)
    cv2.createTrackbar('mask_lower', 'Variable Values',0, 255, nothing)
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    blur_val = 11
    wait = 25
    mask_hsv = 20
    lower_mask = 0

    cycle = 1
    calibrated_x = calibrated_y = calibrated_h = calibrated_w = 0
    if os.path.exists("test_data/"):
        shutil.rmtree("test_data/")
        
    if not os.path.exists("test_data/"):
        os.makedirs("test_data/")


    cap = cv2.VideoCapture(filename)
    
    while(cap.isOpened()):
        start_time = time.time()

        #Capture frames from the camera
        ret, frame = cap.read()
        if not ret:
           break
        hand = frame
        #Use Haar Cascade to detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw the region of interest 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]

            center_face = (int(x+w/2),int(y+h/2))
            center_face_str = np.array(map(str, center_face))

            #Draw center of face
            cv2.circle(frame, center_face, 7, [100,0,255], 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, 'Face Center', center_face, font, 2, (255,255,255), 2)

        #blur_val = cv2.getTrackbarPos('blur', 'Variable Values')

        #Blur the image to remove noised
        blur = cv2.blur(frame,(blur_val,blur_val))
    #   blur = cv2.GaussianBlur(frame, (blur_val,blur_val), 0)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        #Mask blurred image with Hue values of 0-20.
        #mask_hsv = cv2.getTrackbarPos('mask','Variable Values')
        #lower_mask = cv2.getTrackbarPos('mask_lower','Variable Values')
        mask = cv2.inRange(hsv,np.array([lower_mask,50,50]),np.array([mask_hsv,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Morph the image to get rid of noise.
        #Dilation increases the noise while erosion erases noise that was enlarged.
        dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1) 
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        
        #Find contours of the filtered frame
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Find largest contour on the frame
        max_area=100
        i=0	
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                i = n  
                
        if len(contours) > 0:        
            cnts = contours[i]
        else:
            continue

        #Find convex hull of hand contour
        hull = cv2.convexHull(cnts)
        
        #Find convex defects of the hand
        hull2 = cv2.convexHull(cnts, returnPoints = False)
        defects = cv2.convexityDefects(cnts, hull2)
        
        #Get defect points and draw them in the frame
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            cv2.circle(frame,far,5,[150,255,255],2)

        #Find center of hand
        moments = cv2.moments(cnts)
        if moments['m00']!=0:
            x = int(moments['m10']/moments['m00'])
            y = int(moments['m01']/moments['m00'])
        center=(x,y)    

        #Draw center mass
        cv2.circle(frame,center,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,'Center',tuple(center),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            center = np.array(center)
            distance = np.sqrt(np.power(x[0]-center[0],2)+np.power(x[1]-center[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []

        for i in range(0,len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0]-center[0],2)+np.power(fingers[i][1]-center[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0

        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1

        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        if cycle == 1:
            calibrated_x = x
            calibrated_y = y
            calibrated_h = h
            calibrated_w = w

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,255,255),2)
        
        ##### Show final image ########
        filename_exploded = filename.split(".")[0].split("/")
        outfilename = filename_exploded[len(filename_exploded)-1]

        #cv2.imshow('Hand_Mask', mask[calibrated_y:calibrated_y+640, calibrated_x:calibrated_x+360])

        if calibrated_y+640 > mask.shape[0]:
            calibrated_y = mask.shape[0]-640 

        cv2.imwrite("test_data/frame"+str(cycle)+".jpg",  mask[calibrated_y:calibrated_y+640, calibrated_x:calibrated_x+360])

        cycle = cycle + 1
        #close the output video by pressing 'ESC'
        #wait = cv2.getTrackbarPos('wait','Variable Values')
        k = cv2.waitKey(wait) & 0xFF
        if k == 27:
            break
            """
        if k == ord('p'):
            while True:
                k2 = cv2.waitKey(1) or 0xff
                cv2.imshow('Final', frame)
                if k2 == ord('p'):
                    break
            cv2.imshow('Final', frame)"""
           

    cap.release()
    cv2.destroyAllWindows()
