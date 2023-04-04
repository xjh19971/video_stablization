import numpy as np
import cv2
from utils import parse_arguments
import logging
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Inspired by https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
# Adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def KLT(args, prev_gray, frame_gray, p0, img=None):
    # Calculate optical flow for a sparse feature set (Lucas-Kanade Method)
    lk_params = dict(winSize  = args.winSize,
                    maxLevel = args.maxLevel,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    img0, img1 = prev_gray, frame_gray
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

    logging.debug(p1.shape)
    logging.debug(st1.shape)
    # Get all the trajectories
    if img is not None:
        for (x, y) in zip(p1.reshape(-1, 2)):
            # Newest detected point
            if st1:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Draw Lines
    if img is not None:
        for (x, y) in zip(p1.reshape(-1, 2)):
            cv2.polylines(img, [np.int32(p1)], False, (0, 255, 0))
            cv2.putText(img, 'track count: %d' % len(p1), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return p1, st1, img

# GFTT adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def feature_extract(args, prev_gray, mask, img=None, extractor=None):
    if args.feat_ext == "GFTT":
        feature_params = dict(maxCorners = args.maxCorners,
                        qualityLevel = args.qualityLevel,
                        minDistance = args.minDistance,
                        blockSize = args.blockSize)
    elif args.feat_ext == "SIFT":
        feature_params = dict()
    # Detect the good features to track
    if args.feat_ext == "GFTT":
        features = cv2.goodFeaturesToTrack(prev_gray, mask = mask, **feature_params)
        p0 = features.reshape(-1, 2)
    elif args.feat_ext == "SIFT":
        keypoints, descriptors = extractor.detectAndCompute(prev_gray)
    return p0

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
    s = np.array([np.sqrt(M[0, 0]**2 + M[0, 1]**2), np.sqrt(M[1, 0]**2 + M[1, 1]**2)]) # scaling component
    r = np.arctan2(M[1, 0] , M[1, 1]) # rotation component
    transforms.append(np.array([t[0], t[1], s[0], s[1], r]))
    return transforms

def calc_stab_M(args, transforms):
    transforms_np = np.array(transforms)
    trajectory_np = np.cumsum(transforms_np, axis=0)
    trajectory_np_stab = savgol_filter(trajectory_np, window_length=args.smooth_win_len, polyorder=args.smooth_order, axis=0)
    if args.debug:
        plt.subplot(121)
        plt.plot(trajectory_np_stab[:, 0])
        plt.subplot(122)
        plt.plot(trajectory_np[:, 0])
        plt.savefig("t0.png")
        plt.subplot(121)
        plt.plot(trajectory_np_stab[:, 1])
        plt.subplot(122)
        plt.plot(trajectory_np[:, 1])
        plt.savefig("t1.png")
    diff = trajectory_np_stab - trajectory_np
    transforms_np_stab = transforms_np + diff
    stab_M_list = []
    t0_stab = transforms_np_stab[:, 0]
    t1_stab = transforms_np_stab[:, 1]
    s0_stab = transforms_np_stab[:, 2]
    s1_stab = transforms_np_stab[:, 3]
    r_stab = transforms_np_stab[:, 4]
    for i in range(len(t0_stab)):
        stab_M_list.append(np.array([[s0_stab[i] * np.cos(r_stab[i]), -s0_stab[i] * np.sin(r_stab[i]), t0_stab[i]],
                            [s1_stab[i] * np.sin(r_stab[i]), s1_stab[i] * np.cos(r_stab[i]), t1_stab[i]]]))
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

        # Feature extraction
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        if args.feat_ext != "GFTT" and extractor is None:
            extractor = cv2.SIFT_create()
        p0 = feature_extract(args, frame_gray, mask, img, extractor=extractor)

        # Optical Flow
        p1, st1, img = KLT(args, prev_gray, frame_gray, p0, img)

        # Calculate transforms
        if frame_idx > 0:
            M = calc_transformation(p1, st1)
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
    logging.basicConfig(level = logging.DEBUG if args.debug else logging.INFO)
    transforms = extract_transforms(args)
    stab_M_list = calc_stab_M(args, transforms)
    stablize_video(args, stab_M_list)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()