
# How it works:
# You need to introduce the x_data points and y_data points for each particle in the main loop.
# Check "x_data" and "y_data"
# Also you need to introduce the size of the droplet centered in the middle. "radius"
# You need to define a distance from the perimeter of the droplet at which you consider the particle
# has reached the droplet. Check "error".
# You need the video to edit. Check "video_name_original". Add also its fps in "fps".
# Also check "fourcc" and "video_name" first appearing for video output options
# You need to do the main loop as many times as particles you have. Modify "num_particles".
# These modifications have ..... where you need to introduce your data.

import os
import glob
import cv2
import numpy as np
from scipy import stats
import math

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
	"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/math.pi

# Video to edit
video_name_original = .....
fps = .....

# Radius of the droplet in pixels
radius = .....

# Distance at which we start considering contact to droplet
error = .....

# Number of particles in this video
num_particles = .....

# Do the same for each particle
for p in range(0,num_particles):
  # Data for x,y position for this particle
  x_data = [.....]
  y_data = [.....]

  # Properties for output video
  fourcc = cv2.VideoWriter_fourcc(*'FMP4')#MPJG
  video_name = video_name_original.replace('.avi', '_modified.avi')

  # Open new video
  cap = cv2.VideoCapture(video_name)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  # NOTE:
  # fps = cap.get(cv2.CAP_PROP_FPS) do not always work...
  height2 = height/2
  width2 = width/2
  out = cv2.VideoWriter(video_name,fourcc, fps, (int(width), int(height)))
  cap.set(1,0) # Start in the initial frame 0
  frame_now = 0

  # Properties to be calculated
  init_time=[-1,-1] # [contact time x0, incoming time x1]       O-x0-----x1, O is droplet, particle moves from right to left
  final_time=[-1,-1] # [end contact time x0, outgoing time x1]  x1-----x0-O, O is droplet
  slope = [float('nan'), float('nan'), 0, 0]
  intercept = [float('nan'), float('nan'), 0, 0]
  r_value = [0, 0, 0, 0]
  p_value = [0, 0, 0, 0]
  std_err = [0, 0, 0, 0]

  # For each frame
  while(cap.isOpened()) and count < len(x_data):
    # Read the frame
    ret, frame = cap.read()

    # If there's no frame, exit
    if hasattr(frame, 'shape') == False:
      break

    # Obtain info from the frame and copy this frame
    rows,cols,d = frame.shape
    dst = frame

    # Calculate distance of particle to center of droplet
    distance = math.sqrt((x_data[frame_now]-width2)*(x_data[frame_now]-width2) + (y_data[frame_now]-height2)*(y_data[frame_now]-height2))

    # Calculate incoming approximation
    #	We consider a distance of 2*error for the first approximation to calculate incoming/outgoing directions
    # Obtain first time and print always from so on a cross in the video
    if init_time[0]<0 and init_time[1]<0 and distance<=(radius+2*error):
      init_time[1] = count
      dst = cv2.drawMarker(frame, (int(x_data[init_time[1]]),int(y_data[init_time[1]])), (210,6,253), cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    elif init_time[1]>-1:
      dst = cv2.drawMarker(frame, (int(x_data[init_time[1]]),int(y_data[init_time[1]])), (210,6,253), cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    # We consider a distance of error to calculate contact of particle with droplet
    if init_time[0]<0 and distance<=(radius+error):
      init_time[0] = count

      # Do linear fits!
      if init_time[0]-init_time[1] > 1:
        slope[0], intercept[0], r_value[0], p_value[0], std_err[0] = stats.linregress(x_data[init_time[1]:init_time[0]],y_data[init_time[1]:init_time[0]])
        if math.isnan(float(slope[0])) == False and math.isnan(float(intercept[0])) == False:
          dst = cv2.line(dst, (int(x_data[init_time[0]]),int(slope[0]*x_data[init_time[0]]+intercept[0])), (int(x_data[init_time[1]]),int(slope[0]*x_data[init_time[1]]+intercept[0])), (2,106,253), thickness=2)
      else:
        slope[0] = float('nan')
        intercept[0] = float('nan')

      # Print the radius!
      dst = cv2.line(dst, (int(x_data[init_time[0]]),int(y_data[init_time[0]])), (int(width2),int(height2)), (97,19,54), thickness=2)
      dst = cv2.drawMarker(dst, (int(x_data[init_time[0]]),int(y_data[init_time[0]])), (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
    # If we already contacted it, draw elements
    elif init_time[0]>-1:
      dst = cv2.drawMarker(dst, (int(x_data[init_time[0]]),int(y_data[init_time[0]])), (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
      if math.isnan(float(slope[0])) == False and math.isnan(float(intercept[0])) == False:
        if init_time[0]-init_time[1] > 1:
          dst = cv2.line(dst, (int(x_data[init_time[0]]),int(slope[0]*x_data[init_time[0]]+intercept[0])), (int(x_data[init_time[1]]),int(slope[0]*x_data[init_time[1]]+intercept[0])), (2,106,253), thickness=2)

      dst = cv2.drawMarker(dst, (int(x_data[init_time[0]]),int(y_data[init_time[0]])), (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
      dst = cv2.line(dst, (int(x_data[init_time[0]]),int(y_data[init_time[0]])), (int(width2),int(height2)), (97,19,54), thickness=2)

    # Calculate outgoing approximation
    # Same scheme as before
    if final_time[0]>-1 and final_time[1]<0 and distance>=(radius+2*error):
      final_time[1] = count-1
      dst = cv2.drawMarker(dst, (int(x_data[final_time[1]]),int(y_data[final_time[1]])), (210,6,253), cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    elif final_time[1]>-1:
      dst = cv2.drawMarker(dst, (int(x_data[final_time[1]]),int(y_data[final_time[1]])), (210,6,253), cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    if final_time[0]<0 and init_time[0]>-1 and distance>=(radius+error):
      final_time[0] = count-1
      dst = cv2.drawMarker(dst, (int(x_data[final_time[0]]),int(y_data[final_time[0]])), (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
      dst = cv2.line(dst, (int(x_data[final_time[0]]),int(y_data[final_time[0]])), (int(width2),int(height2)), (97,19,54), thickness=2)
    elif final_time[0]>-1 and init_time[0]>-1:
      dst = cv2.drawMarker(dst, (int(x_data[final_time[0]]),int(y_data[final_time[0]])), (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
      dst = cv2.line(dst, (int(x_data[final_time[0]]),int(y_data[final_time[0]])), (int(width2),int(height2)), (97,19,54), thickness=2)

      # Calculate the turns manually. You need to introduce the number of turns it did.
      # It must be complete turns or, a complete turns and a half. Then the code will find the exact reminder after that
      turns_manual = float(input("How many turns did the particle around the droplet? Indicate integer or integer.5\n"))
      #Now I calculate how much over 180º is the left
      rad_line_1 = [x_data[init_time[0]]-width2, y_data[init_time[0]]-height2]
      rad_line_2 = [x_data[final_time[0]]-width2, y_data[final_time[0]]-height2]

      #Check if we pass 1/2 of the droplet. If not, the angle to calculate is between the entry and the exit
      #If it has passed, it must be between the exit and the opposite "radius" to the entrance one.
      if int(2.*turns_manual) % 2 == 0:
        turns_manual += angle_between(rad_line_1, rad_line_2)/360
        print (angle_between(rad_line_1, rad_line_2))
      else:
        turns_manual += (180-angle_between(rad_line_1, rad_line_2))/360
        print (180-angle_between(rad_line_1, rad_line_2))

    #Do linear fits!
    if final_time[1]-final_time[0] > 1:
      if math.isnan(slope[1]):
        slope[1], intercept[1], r_value[1], p_value[1], std_err[1] = stats.linregress(x_data[final_time[0]:final_time[1]],y_data[final_time[0]:final_time[1]])
      if math.isnan(float(slope[1])) == False and math.isnan(float(intercept[1])) == False:
        dst = cv2.line(dst, (int(x_data[final_time[0]]),int(slope[1]*x_data[final_time[0]]+intercept[1])), (int(x_data[final_time[1]]),int(slope[1]*x_data[final_time[1]]+intercept[1])), (2,106,253), thickness=2)
    else:
      slope[1] = float('nan')
      intercept[1] = float('nan')

    cv2.imshow('frame',dst)
    out.write(dst)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Close video
  cap.release()
  out.release()
  cv2.destroyAllWindows()

  # Now we calculate turns
  # radius lines
  slope[2], intercept[2], r_value[2], p_value[2], std_err[2] = stats.linregress([x_data[init_time[0]], width2],[y_data[init_time[0]], height2])
  slope[3], intercept[3], r_value[3], p_value[3], std_err[3] = stats.linregress([x_data[final_time[0]], width2],[y_data[final_time[0]], height2])

  # I need the perpendicular line!!!!
  slope[2] = -1/slope[2]
  slope[3] = -1/slope[3]

  angles = [float('nan'), float('nan'), float('nan')]

  # We correct possible nans for incoming particle
  # both are nans => 90º!!!!
  if math.isnan(slope[0]) and math.isnan(slope[2]):
    angles[0] = 90.
  # one is nan
  elif math.isnan(slope[0]) or math.isnan(slope[2]):
    # correct the one needed
    if math.isnan(slope[0]):
      slope[0] = 0
      slope[2] = -1/slope[2]
      # just check in case!
      if math.isnan(slope[2]):
        angles[0] = 90
      else:
        angles[0] = math.atan(math.fabs((slope[0]-slope[2])/(1+slope[0]*slope[2])))*180/math.pi
    else:
      slope[2] = 0
      slope[0] = -1/slope[0]
      # just check in case!
      if math.isnan(slope[0]):
        angles[0] = 90
      else:
        angles[0] = math.atan(math.fabs((slope[0]-slope[2])/(1+slope[0]*slope[2])))*180/math.pi
  else:
    angles[0] = math.atan(math.fabs((slope[0]-slope[2])/(1+slope[0]*slope[2])))*180/math.pi

  # Now for exiting particles!
  if math.isnan(slope[1]) and math.isnan(slope[3]):
    angles[1] = 90.
  #one is nan
  elif math.isnan(slope[1]) or math.isnan(slope[3]):
    #correct the one needed
    if math.isnan(slope[1]):
      slope[1] = 0
      slope[3] = -1/slope[3]
      #just check in case!
      if math.isnan(slope[3]):
        angles[1] = 90
      else:
        angles[1] = math.atan(math.fabs((slope[1]-slope[3])/(1+slope[1]*slope[3])))*180/math.pi
    else:
      slope[3] = 0
      slope[1] = -1/slope[1]
      #just check in case!
      if math.isnan(slope[1]):
        angles[1] = 90
      else:
        angles[1] = math.atan(math.fabs((slope[1]-slope[3])/(1+slope[1]*slope[3])))*180/math.pi
  else:
    angles[1] = math.atan(math.fabs((slope[1]-slope[3])/(1+slope[1]*slope[3])))*180/math.pi

  slope[2] = -1/slope[2]
  slope[3] = -1/slope[3]

  rad_line_1 = [x_data[init_time[0]]-width2, y_data[init_time[0]]-height2]
  rad_line_2 = [x_data[final_time[0]]-width2, y_data[final_time[0]]-height2]

  angles[2] = angle_between(rad_line_1, rad_line_2)

  # print the values you want.
  # total time in contact (final_time[0]-init_time[0])/fps in seconds
  # drop_size: radius*resolution_pxiel_microns
  # turns: turns_manual
  # 'angle in': angles[0], 'angle out': angles[1], 'angle phi': angles[2], 'angle phi 360': 360-angles[2]
