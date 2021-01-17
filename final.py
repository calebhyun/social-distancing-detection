# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

print("successfully imported")

vidcap = cv2.VideoCapture('people.mp4')

directory = r'C:\Users\caleb\source\repos\opencvtest\splitvid'

#os.chdir(directory)

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,frame = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(directory, "frame"+str(count)+".jpg"), frame)     # save frame as JPG file
    return hasFrames

sec = 0
frameRate = 0.5 # capture image in each 0.5 second
count=1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#imagePath = "store3.jpg"

pointsofintereststart = []
pointsofinterestend = []


imagePath = "splitvid\\frame1.jpg"
image = cv2.imread(imagePath)
image = imutils.resize(image, width=min(400, image.shape[1]))

def onMouse(event, x, y, flags, param):
    global pointsofintereststart, pointsofinterestend
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsofintereststart.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        pointsofinterestend.append((x, y))

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Choose Points')
cv2.setMouseCallback('Choose Points', onMouse)

while True:
    cv2.imshow("Choose Points", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print("pointsofintereststart: ", pointsofintereststart, "pointsofinterestend: ", pointsofinterestend)

peoplecoords = []

color = (0, 0, 0)

for imagePath in paths.list_images(directory):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    beforeimage = cv2.imread(imagePath)
    beforeimage = imutils.resize(image, width=min(400, image.shape[1]))
    
    print("got into for imagepath")
    for i in range(0, len(pointsofintereststart)):
        print("got into poi for loop")
        image = beforeimage[pointsofintereststart[i][1]:pointsofinterestend[i][1], pointsofintereststart[i][0]:pointsofinterestend[i][0]].copy()
        #print("1", pointsofintereststart[i][1], "2", pointsofinterestend[i][1], "3", pointsofintereststart[i][0], "4", pointsofinterestend[i][0])
        orig = image.copy()
        #cv2.imshow("After NMS", image)  
        #cv2.waitKey(0)

        print("past orig")
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=.8) #og: 1.05

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            avgcoordsx = x + w/2
            avgcoordsy = y + h/2
            peoplecoords.append((avgcoordsx, avgcoordsy))
            print("peoplecoords:", peoplecoords)
        
        
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=.65) #og: .65

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)


        # show some information on the number of bounding boxes
        filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
        
        print("right before cv2.imshow")
        cv2.imshow("After NMS", image)  
        cv2.waitKey(0)

        peoplecoords.clear() 


#for i in range(0, len(pointsofinterestend)):
#        image = cv2.rectangle(image, pointsofintereststart[i], pointsofinterestend[i], color, 3)
        #for j in range(0, len(peoplecoords)):
            #if pointsofinterestend[i][0] < peoplecoords[j][0] and pointsofintereststart[i][0] > peoplecoords[j][0] and pointsofinterestend[i][1] < peoplecoords[j][1] and pointsofintereststart[i][1] < peoplecoords[j][1]:
