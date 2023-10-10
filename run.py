#!/usr/bin/env python3
"""Main executable module"""
from __future__ import annotations

from extensions import *
from tracker.centroid import CentroidTracker as Centroid
from tracker.track import TrackableObject as TrackObj
from utils.helper import parse_arguments


__author__ = "Krupesh1958 | Arjun-234"
__version__ = 0.1


args = parse_arguments()

# Load serialized model from disk with `Deep neural network`
load_model = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# If video path not supplied, grab a reference to ip camera,
# else grab a reference to the video file
if not args.get("input"):
	logger.info("Starting the live stream...")
	video_stream = VideoStream(args["url"]).start()
	time.sleep(2)
else:
	logger.info("Starting the video...")
	video_stream = cv.VideoCapture(args["input"])

# Instantiate our centroid, then initialize a list to store
# each of our dlib correlation trackers
centroid_tracker = Centroid(max_disappeared=40, max_distance=50)

while True:
	# Grab the next frame and handle if we are reading from either
	# videoCapture or videoStream
	frame = video_stream.read()
	frame = frame[1] if args.get("input", False) else frame

	if args["input"] is not None and frame is None:
		break

	# Resize the frame to have a maximum width of 500p, then convert
	# the frame from BGR to RGB for dlib
	frame = resize(frame, width=500)
	rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

	if not width or not height:
		height, width = frame.shape[:2]


	status = "Waiting"
	rects = []

	# Detecting or Tracking
	if total_frames % args["skip_frames"] == 0:
		trackers = []

		# Frame to blob and obtain the detections
		blob = cv.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
		load_model.setInput(blob)
		detections = load_model.forward()

		# loop over the detections
		for itr in np.arange(0, detections.shape[2]):
			configdence = detections[0, 0, itr, 2]
			if configdence > args["confidence"]:
				idx = int(detections[0, 0, itr, 1])
				if idx != 15:
					continue
				box = detections[0, 0, itr, 3:7] * np.array([width, height, width, height])
				(start_x, start_y, end_x, end_y) = box.astype("int")

				# Construct a dlib rectangle object from the bounding
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(start_x, start_y, end_x, end_y)
				tracker.start_track(rgb, rect)
				trackers.append(tracker)

	else:
		for tracker in trackers:
			status = "Tracking"

			tracker.update(rgb)
			pos = tracker.get_position()

			start_x = int(pos.left())
			start_y = int(pos.top())
			end_x = int(pos.right())
			end_y = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((start_x, start_y, end_x, end_y))


	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv.line(
		img=frame,
		pt1=(0, height-100 // 2),
		pt2=(width, height-100 // 2),
		color=(0, 0, 0),
		thickness=3
	)
	objects = centroid_tracker.update(rects)

	for (object_id, centroid) in objects.items():
		to = trackable_objects.get(object_id, None)

		if not to:
			to = TrackObj(object_id, centroid)
		else:
			# The difference between the y-coordinate of the `current`
			# centroid and the mean of `previous` centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y_coordinate = [itr[1] for itr in to.centroids]
			direction = centroid[1] - np.mean(y_coordinate)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				if direction < 0 and centroid[1] < height // 2:
					total_up += 1
					date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
					move_out.append(total_up)
					out_time.append(date_time)
					to.counted = True

				elif direction > 0 and centroid[1] > height // 2:
					total_down += 1
					date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
					move_in.append(total_down)
					in_time.append(date_time)
					to.counted = True
					# compute the sum of total people inside
					total = [len(move_in) - len(move_out)]


		# store the trackable object in our dictionary
		trackable_objects[object_id] = to

	info_status = [
		("Status", status),
		("Count", (total_down - total_up))
	]

	for (itr, (key, value)) in enumerate(info_status):
		text = "%s: %s" % (key, value)
		cv.putText(
			img=frame,
			text=text,
			org=(10, height - ((itr * 20) + 20)),
			fontFace=cv.FONT_HERSHEY_SIMPLEX,
			fontScale=0.6,
			color=(0, 0, 0),
			thickness=2
		)

	if writer is not None:
		writer.write(frame)

	cv.imshow("Real-Time Monitoring/Analysis Window", frame)
	key = cv.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# Increment total number of frames
	total_frames += 1

cv.destroyAllWindows()
