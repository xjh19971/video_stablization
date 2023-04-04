import numpy as np
import cv2
from utils import parse_arguments
import logging
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
logging.basicConfig(level = logging.DEBUG)

# Inspired by https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
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
    valid_trajs_idxs = np.where(st_np == 1)[0]
    # valid_trajs_idxs = np.arange(len(trajs))
    curr = []
    prev = []
    for idx in valid_trajs_idxs:
        if len(trajs[idx]) >= 2:
            curr.append(trajs[idx][-1])
            prev.append(trajs[idx][-2])
    [M, inliers] = cv2.estimateAffinePartial2D(np.array(curr), np.array(prev)) # Partial affine to only contains rotation, translation and scaling
    return M

def calc_transforms(args, M, transforms):
    # logging.debug(M)
    t = M[:, 2] # translation component
    # s = np.array([np.sqrt(M[0, 0]**2 + M[0, 1]**2), np.sqrt(M[1, 0]**2 + M[1, 1]**2)]) # scaling component
    r = np.arctan2(M[1, 0] , M[1, 1]) # rotation component
    # transforms.append(np.array([t[0], t[1], s[0], s[1], r]))
    transforms.append(np.array([t[0], t[1], r]))
    return transforms

def calc_stab_M(args, transforms):
    transforms_np = np.array(transforms)
    trajectory_np = np.cumsum(transforms_np, axis=0)
    trajectory_np_stab = savgol_filter(trajectory_np, window_length=args.smooth_win_len, polyorder=args.smooth_order, axis=0)
    plt.subplot(121)
    plt.plot(trajectory_np_stab[:, 1])
    plt.subplot(122)
    plt.plot(trajectory_np[:, 1])
    plt.savefig("temp.png")
    diff = trajectory_np_stab - trajectory_np
    transforms_np_stab = transforms_np + diff
    stab_M_list = []
    t0_stab = transforms_np[:, 0]
    t1_stab = transforms_np[:, 1]
    r_stab = transforms_np[:, 2]
    # t0_stab = transforms_np_stab[:, 0]
    # t1_stab = transforms_np_stab[:, 1]
    # # s0_stab = transforms_np_stab[:, 2]
    # # s1_stab = transforms_np_stab[:, 3]
    # # r_stab = transforms_np_stab[:, 4]
    # r_stab = transforms_np_stab[:, 2]
    for i in range(len(t0_stab)):
        # stab_M_list.append(np.array([[s0_stab[i] * np.cos(r_stab[i]), -s0_stab[i] * np.sin(r_stab[i]), t0_stab[i]],
        #                     [s1_stab[i] * np.sin(r_stab[i]), s1_stab[i] * np.cos(r_stab[i]), t1_stab[i]]]))
        stab_M_list.append(np.array([[np.cos(r_stab[i]), -np.sin(r_stab[i]), t0_stab[i]],
                            [np.sin(r_stab[i]), np.cos(r_stab[i]), t1_stab[i]]]))
    stab_M_list = np.stack(stab_M_list, axis=0)
    return stab_M_list

def stablize(args, prev_frame, M_stab):
    rows, cols, c = prev_frame.shape
    frame_new = cv2.warpAffine(prev_frame, M_stab, (cols, rows))
    return frame_new

def resize_center_crop(args, frame):
    rows, cols, c = frame.shape
    resized = cv2.resize(frame, (cols*2, rows*2))
    center_x = cols
    center_y = rows
    left = center_x - cols // 2
    top = center_y - rows // 2
    right = center_x + cols // 2
    bottom = center_y + rows // 2
    cropped = resized[top:bottom, left:right]
    return cropped

def extract_transforms(args):
    detect_interval = args.DetectInterval
    trajs = []
    st = []
    frame_idx = 0   
    cap = cv2.VideoCapture(os.path.join(args.data_folder, args.dataset, args.type, args.video_name))
    prev_gray = None
    extractor = None
    transforms = []
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

        # Calculate transforms
        if frame_idx > 0:
            M = calc_transformation(trajs, st)
            transforms = calc_transforms(args, M, transforms)
        
        frame_idx += 1
        prev_gray = frame_gray
        
        # Show Results
        if img is not None:
            cv2.imshow('Optical Flow', img)
            cv2.imshow('Mask', mask)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    return transforms

def stablize_video(args, stab_M_list):
    frame_idx = 0   
    cap = cv2.VideoCapture(os.path.join(args.data_folder, args.dataset, args.type, args.video_name))
    prev_frame = None
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame -- or Reached end of video.. Exiting ...")
            break

        # Stablize video
        if frame_idx > 0:
            frame_stab = stablize(args, prev_frame, stab_M_list[frame_idx-1])
        else:
            frame_stab = frame
        
        frame_idx += 1
        prev_frame = frame
        
        # Show Results
        cv2.imshow('Unstablized', resize_center_crop(args,frame))
        cv2.imshow('Stablized', resize_center_crop(args,frame_stab))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

def main():
    args = parse_arguments()
    transforms = extract_transforms(args)
    stab_M_list = calc_stab_M(args, transforms)
    stablize_video(args, stab_M_list)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()