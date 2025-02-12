import cv2 as cv
from tracker import *


tracker = EuclideanDistTracker()

cap = cv.VideoCapture("los_angeles.mp4")
object_detector = cv.createBackgroundSubtractorMOG2(history = 100, varThreshold = 45)

while True:
	ret, frame = cap.read()
	roi = frame[340: 720, 500: 800]
	mask = object_detector.apply(frame)
	_, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
	contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	detections = []

	for cnt in contours:
		area = cv.contourArea(cnt)
		if area > 350:
			x, y, w, h = cv.boundingRect(cnt)
			detections.append([x, y, w, h])
			#cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
			#cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
		#print(detections)


	b_ids = tracker.update(detections)

	for b_id in b_ids:
		x, y, w, h, id = b_id
		#cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
		cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
		#cv.putText(frame, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	cv.imshow("Frame", frame)
	cv.imshow("ROI", roi)
	#cv.imshow("Mask", mask)

	key = cv.waitKey(1)
	if key == 27:
		break

cap.release()
cv.destroyAllWindows()

	
