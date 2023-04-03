import numpy as np
import cv2 as cv
from utils import parse_arguments

# Adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def LKT(args, prev_gray, frame_gray, trajs, img=None):
    # Calculate optical flow for a sparse feature set (Lucas-Kanade Method)
    lk_params = dict(winSize  = args.winSize,
                    maxLevel = args.maxLevel,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    traj_len = args.trajLen # make larger if we want to track for longer
    
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
            if img != None:
                cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajs = new_trajs

        # Draw Lines
        if img!= None:
            cv.polylines(img, [np.int32(traj) for traj in trajs], False, (0, 255, 0))
            cv.putText(img, 'track count: %d' % len(trajs), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return traj, img

def feature_extract(frame_gray, trajs, img=None):
    mask = np.zeros_like(frame_gray)
    mask[:] = 255

    # Lastest point in latest trajectory
    if img != None:
        for x, y in [np.int32(traj[-1]) for traj in trajs]:
            cv.circle(mask, (x, y), 5, 0, -1)

    # Detect the good features to track
    features = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
    if features is not None:
        # If good features can be tracked - append to trajs
        for x, y in np.float32(features).reshape(-1, 2):
            trajs.append([(x, y)])
    
    return trajs

def main():
    args = parse_arguments()
    # Parameters for Optical Flow

    # Parameters for Corner Detection
    feature_params = dict(maxCorners = args.maxCorners,
                        qualityLevel = args.qualityLevel,
                        minDistance = args.minDistance,
                        blockSize = args.blockSize)

    detect_interval = args.DetectInterval
    trajs = []
    frame_idx = 0
    # Change name of video input
    cap = cv.VideoCapture('1.avi')

    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame -- or Reached end of video.. Exiting ...")
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if args.visualize:
            img = frame.copy()
        else:
            img = None

        trajs, img = LKT(args, prev_gray, frame_gray, trajs, img)
        # Update interval - When to update and detect new features
        if frame_idx % detect_interval == 0:
            trajs, mask = feature_extract(frame_gray, trajs, img)

        frame_idx += 1
        prev_gray = frame_gray
        
        # Show Results
        cv.imshow('Optical Flow', img)
        cv.imshow('Mask', mask)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()