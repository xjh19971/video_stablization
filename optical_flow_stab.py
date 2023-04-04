import numpy as np
import cv2
from utils import parse_arguments
import logging
import os
from scipy.interpolate import splrep, BSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
logging.basicConfig(level = logging.DEBUG)

# Adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def KLT(args, prev_gray, frame_gray, trajs, st, img=None):
    # Calculate optical flow for a sparse feature set (Lucas-Kanade Method)
    lk_params = dict(winSize  = args.winSize,
                    maxLevel = args.maxLevel,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    traj_len = args.trajLen # make larger if we want to track for longer
    
    if len(trajs) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([traj[-1][0:2] for traj in trajs]).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        difference = abs(p0-p0r).reshape(-1, 2).max(-1)
        good_detect = difference < 1 # Why does it requires difference < 1?

        new_trajs = []
        new_st = []

        # Get all the trajectories
        for traj, (x, y), good_flag, st1_single in zip(trajs, p1.reshape(-1, 2), good_detect, st1):
            if not good_flag:
                continue
            traj.append((x, y))
            new_st.append(st1_single[0])
            if len(traj) > traj_len:
                del traj[0]
            new_trajs.append(traj)
            # Newest detected point
            if img is not None:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajs = new_trajs
        st = new_st

        # Draw Lines
        if img is not None:
            cv2.polylines(img, [np.int32(traj) for traj in trajs], False, (0, 255, 0))
            cv2.putText(img, 'track count: %d' % len(trajs), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return trajs, st, img

# GFTT adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def feature_extract(args, frame_gray, mask, trajs, st, img=None, extractor=None):
    if args.feat_ext == "GFTT":
        feature_params = dict(maxCorners = args.maxCorners,
                        qualityLevel = args.qualityLevel,
                        minDistance = args.minDistance,
                        blockSize = args.blockSize)
    elif args.feat_ext == "SIFT":
        feature_params = dict()
    # Lastest point in latest trajectory
    if img is not None:
        for x, y in [np.int32(traj[-1]) for traj in trajs]:
            cv2.circle(mask, (x, y), 5, 0, -1)

    # Detect the good features to track
    if args.feat_ext == "GFTT":
        features = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
    elif args.feat_ext == "SIFT":
        keypoints, descriptors = extractor.detectAndComputer(frame_gray)
    if features is not None:
        # If good features can be tracked - append to trajs
        if args.feat_ext == "GFTT":
            for x, y in np.float32(features).reshape(-1, 2):
                trajs.append([(x, y)])
                st.append(1)
        elif args.feat_ext == "SIFT":
            for x, y in np.float32(features).reshape(-1, 2):
                trajs.append([(keypoints, descriptors)])
                st.append(1)
    
    return trajs, st, mask

def calc_transformation(trajs, st):
    st_np = np.array(st)
    # valid_trajs_idxs = np.where(st_np == 1)[0]
    valid_trajs_idxs = np.arange(len(trajs))
    curr = []
    prev = []
    for idx in valid_trajs_idxs:
        if len(trajs[idx]) >= 2:
            curr.append(trajs[idx][-1])
            prev.append(trajs[idx][-2])
    [M, inliers] = cv2.estimateAffinePartial2D(np.array(curr), np.array(prev)) # Partial affine to only contains rotation, translation and scaling
    return M

def calc_stab_M(args, M, states):
    # logging.debug(M)
    t = M[:, 2] # translation component
    s = np.array([np.sqrt(M[0, 0]**2 + M[0, 1]**2), np.sqrt(M[1, 0]**2 + M[1, 1]**2)]) # scaling component
    r = np.arctan2(M[1, 0] , M[1, 1]) # rotation component
    states.append(np.array([t[0], t[1], s[0], s[1], r]))
    # Beizer curve fitting for each components
    states_np = np.array(states)
    if len(states_np) <= 30:
        stab_M = M
    else:
        t0_stab = savgol_filter(states_np[:, 0], window_length=30, polyorder=3)[-1]
        t1_stab = savgol_filter(states_np[:, 1], window_length=30, polyorder=3)[-1]
        s0_stab = savgol_filter(states_np[:, 2], window_length=30, polyorder=3)[-1]
        s1_stab = savgol_filter(states_np[:, 3], window_length=30, polyorder=3)[-1]
        r_stab = savgol_filter(states_np[:, 4], window_length=30, polyorder=3)[-1]
        if len(states_np) == 50:
            plt.subplot(121)
            plt.plot(savgol_filter(states_np[:, 0], window_length=30, polyorder=3))
            plt.plot(states_np[:, 0])
            plt.savefig("temp.png")
        stab_M = np.array([[s0_stab * np.cos(r_stab), -s0_stab * np.sin(r_stab), t0_stab],
                        [s1_stab * np.sin(r_stab), s1_stab * np.cos(r_stab), t1_stab]])
    return stab_M, states

def stablize(args, frame_gray, prev_frame, trajs, st, states):
    M = calc_transformation(trajs, st)
    M_stab, states = calc_stab_M(args, M, states)
    rows, cols, c = prev_frame.shape
    frame_new = cv2.warpAffine(prev_frame, M_stab, (cols, rows))
    return frame_new, states

    
    
def main():
    args = parse_arguments()
    detect_interval = args.DetectInterval
    trajs = []
    st = []
    frame_idx = 0
    # Change name of video input
    cap = cv2.VideoCapture(os.path.join(args.data_folder, args.dataset, args.type, args.video_name))
    prev_gray = None
    prev_frame = None
    extractor = None
    states = []
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame -- or Reached end of video.. Exiting ...")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.visualize:
            img = frame.copy()
        else:
            img = None

        # Optical Flow
        trajs, st, img = KLT(args, prev_gray, frame_gray, trajs, st, img)

        # Feature extraction
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            if args.feat_ext != "GFTT" and extractor is None:
                extractor = cv2.SIFT_create()
            trajs, st, mask = feature_extract(args, frame_gray, mask, trajs, st, img)

        # Calculate transformation
        if frame_idx >= 2:
            stablized_frame, states = stablize(args, frame_gray, prev_frame, trajs, st, states)
        else:
            stablized_frame = frame        
        
        frame_idx += 1
        prev_gray = frame_gray
        prev_frame = frame
        
        # Show Results
        if img is not None:
            cv2.imshow('Optical Flow', img)
            cv2.imshow('Mask', mask)
            cv2.imshow('Stablized', stablized_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()