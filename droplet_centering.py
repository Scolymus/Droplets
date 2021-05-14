
# How it works:
# I have files in a folder called mypath. In windows, the / inside a string is //.
# you can use os.path.sep as a string to define a / either in unix or windows.
# eg. "C:"+os.path.sep+"videos"+os.path.sep
# Change "mypath" with the folder where you have the videos
# You need to introduce the x_data points and y_data points for each droplet.
# In this example we suppose we have only one, but you could have more than one per video
# Modify "x_data" and "y_data" with the x,y positions of your droplets
# Check "x_data" and "y_data"
# Check "fps" and "fourcc" for your video

import os
import glob
import cv2
import csv
import numpy as np

mypath = .....
fps = .....

onlyfiles = glob.glob(mypath+"*.avi")

for filename in onlyfiles:
    # Data for x,y position for one droplet
	x_data = .....
	y_data = .....

    # Properties for output video
	fourcc = cv2.VideoWriter_fourcc(*'FMP4')#MPJG
	video_name = filename.replace('.avi', '_centered.avi')

    # Open new video
	cap = cv2.VideoCapture(video_name)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # NOTE:
    # fps = cap.get(cv2.CAP_PROP_FPS) do not always work...
	out = cv2.VideoWriter(video_name, fourcc, fps, (int(width), int(height)))

	count = 0
	cap.set(1,0) # Start in the initial frame
    # For each frame
	while(cap.isOpened()) and count < len(x_data):
		ret, frame = cap.read()
		rows,cols,d = frame.shape
		M = np.float32([[1,0,cols/2-x_data[count]],[0,1,rows/2-y_data[count]]])
        # In the next line, you could do this transformation better.
        # For example, if your droplet is at x=0, and you center it to width/2
        # Then you will only have half of your video, because from the center to the rigth
        # will be lost. Instead, you should define the new size as cols + width/2 - x_min
        # and move it to the center of that size.
        # If x_min is above the middle, the same will happen but on the other sense.
        # hence it should be cols + abs(width/2 - x_min)
        # Same for y direction. Notice that in OpenCV, y=0 is the top of the video frame,
        # and y=height is the bottom of the video frame
		dst = cv2.warpAffine(frame,M,(cols,rows))
		cv2.imshow('frame',dst)
		out.write(dst)
		count += 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

    # Close the video
	cap.release()
	out.release()
	cv2.destroyAllWindows()
