# USAGE
# python match.py --template cod_logo.png --images images

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", required=True,
                help="Path to template image")
ap.add_argument("-i", "--images", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")


args = vars(ap.parse_args())


# loop over the images to find the template in
for tempPath in glob.glob(args["templates"] + "/*.png"):
    
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(tempPath)
    template = cv2.cvtColor(template, cv2.TM_CCOEFF_NORMED)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    for imagePath in glob.glob(args["images"] + "/*.png"):
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        for scale in np.linspace(0.05, 1.0, 40)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
            (min_val, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            
            print(maxVal, min_val)
            
            # check to see if the iteration should be visualized
            if args.get("visualize", False):
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.waitKey(0)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
                
            #Set minimum confidence level
            minThresh = 0.3
            if maxVal >= minThresh:
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        if found != None:
            (maxVal, maxLoc, r) = found
    
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    
            # draw a bounding box around the detected result and display the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        else:
            print("No Match")


#TO run
#python match.py --template templates --images images --visualize