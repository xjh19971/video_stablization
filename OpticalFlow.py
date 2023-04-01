# Code adapted from: https://github.com/niconielsen32/ComputerVision.git
import numpy as np
import cv2 as cv

# Parameters for Optical Flow
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Corner Detection
feature_params = dict(maxCorners = 100,
                    qualityLevel = 0.3,
                    minDistance = 5,
                    blockSize = 7 )

traj_len = 50 # make larger if we want to track for longer
detect_interval = 10
trajs = []
frame_idx = 0

# Currently using webcam as input, will need to input video from dataset 
cap = cv.VideoCapture(-1)

while(cap.isOpened()):

    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = frame.copy()

    # Calculate optical flow for a sparse feature set (Lucas-Kanade Method)
    if len(trajs) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([traj[-1] for traj in trajs]).reshape(-1, 1, 2)
        p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        difference = abs(p0-p0r).reshape(-1, 2).max(-1)
        good_detect = difference < 1

        new_trajs = []

        # Get all the trajectories
        for traj, (x, y), good_flag in zip(trajs, p1.reshape(-1, 2), good_detect):
            if not good_flag:
                continue
            traj.append((x, y))
            if len(traj) > traj_len:
                del traj[0]
            new_trajs.append(traj)
            # Newest detected point
            cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajs = new_trajs

        # Draw Lines
        cv.polylines(img, [np.int32(traj) for traj in trajs], False, (0, 255, 0))
        cv.putText(img, 'track count: %d' % len(trajs), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(traj[-1]) for traj in trajs]:
            cv.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        features = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if features is not None:
            # If good features can be tracked - append to trajs
            for x, y in np.float32(features).reshape(-1, 2):
                trajs.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray
    
    # Show Results
    cv.imshow('Optical Flow', img)
    cv.imshow('Mask', mask)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()