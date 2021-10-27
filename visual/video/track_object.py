from imutils.video import VideoStream


def track_object():
  # define the lower and upper boundaries of the "green"
  # ball in the HSV color space
  buffer = 32 # frames
  greenLower = (29, 86, 6)
  greenUpper = (64, 255, 255)
  # initialize the list of tracked points, the frame counter,
  # and the coordinate deltas
  pts = deque(maxlen=buffer)
  counter = 0
  (dX, dY) = (0, 0)
  direction = ""
  # if a video path was not supplied, grab the reference
  # to the webcam
  if not args.get("video", False):
    vs = VideoStream(src=0).start()
  # otherwise, grab a reference to the video file
  else:
    vs = cv.VideoCapture(args["video"])
  # allow the camera or video file to warm up
  time.sleep(2.0)





  # resize the frame, blur it, and convert it to the HSV
  # color space
  frame = imutils.resize(frame, width=600)
  blurred = cv.GaussianBlur(frame, (11, 11), 0)
  hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
  # construct a mask for the color "green", then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = cv.inRange(hsv, greenLower, greenUpper)
  mask = cv.erode(mask, None, iterations=2)
  mask = cv.dilate(mask, None, iterations=2)
  # find contours in the mask and initialize the current
  # (x, y) center of the ball
  cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  center = None



  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv.contourArea)
    ((x, y), radius) = cv.minEnclosingCircle(c)
    M = cv.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # only proceed if the radius meets a minimum size
    if radius > 10:
      # draw the circle and centroid on the frame,
      # then update the list of tracked points
      cv.circle(frame, (int(x), int(y)), int(radius),
        (0, 255, 255), 2)
      cv.circle(frame, center, 5, (0, 0, 255), -1)
      pts.appendleft(center)

  # loop over the set of tracked points
  for i in np.arange(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
    if pts[i - 1] is None or pts[i] is None:
      continue
    # check to see if enough points have been accumulated in
    # the buffer
    if counter >= 10 and i == 1 and pts[-10] is not None:
      # compute the difference between the x and y
      # coordinates and re-initialize the direction
      # text variables
      dX = pts[-10][0] - pts[i][0]
      dY = pts[-10][1] - pts[i][1]
      (dirX, dirY) = ("", "")
      # ensure there is significant movement in the
      # x-direction
      if np.abs(dX) > 20:
        dirX = "East" if np.sign(dX) == 1 else "West"
      # ensure there is significant movement in the
      # y-direction
      if np.abs(dY) > 20:
        dirY = "North" if np.sign(dY) == 1 else "South"
      # handle when both directions are non-empty
      if dirX != "" and dirY != "":
        direction = "{}-{}".format(dirY, dirX)
      # otherwise, only one direction is non-empty
      else:
        direction = dirX if dirX != "" else dirY
