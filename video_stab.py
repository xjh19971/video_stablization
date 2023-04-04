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

    # Get all the trajectories
    if img is not None:
        for (x, y), st in zip(p1.reshape(-1, 2), st1):
            # Newest detected point
            if st:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Draw Lines
    if img is not None:
        for (x, y), (x0, y0) in zip(p1.reshape(-1, 2), p0):
            cv2.polylines(img, np.array([np.int32((x0, y0)), np.int32((x, y))]).reshape((-1, 1, 2)), False, (0, 255, 0))
            cv2.putText(img, 'track count: %d' % len(p1), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return p1, st1, img

# GFTT adapted from https://github.com/niconielsen32/ComputerVision/blob/master/opticalFlow/sparseOpticalFlow.py
def feature_extract(args, prev_gray, mask, img=None, extractor=None):
    if args.feat_ext == "GFTT":
        feature_params = dict(maxCorners = args.maxCorners,
                        qualityLevel = args.qualityLevel,
                        minDistance = args.minDistance,
                        blockSize = args.blockSize)
    elif args.feat_ext == "SIFT" or args.feat_ext == "ORB":
        feature_params = dict()
    # Detect the good features to track
    if args.feat_ext == "GFTT":
        features = cv2.goodFeaturesToTrack(prev_gray, mask = mask, **feature_params)
        p0 = features.reshape(-1, 2)
    elif args.feat_ext == "SIFT" or args.feat_ext == "ORB":
        keypoints, descriptors = extractor.detectAndCompute(prev_gray, mask = mask, **feature_params)
        p0 = (keypoints, descriptors)
    return p0

def matching(args, prev_gray, frame_gray, p0, mask, img=None, extractor=None, matcher=None):
    if args.feat_ext == "GFTT":
        p1, st1, img = KLT(args, prev_gray, frame_gray, p0, img)
    elif args.feat_ext == "SIFT" or args.feat_ext == "ORB":
        # BF match: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        new_p0 = []
        p1 = []
        st1 = []
        kp0, desc0 = p0
        kp1, desc1 = extractor.detectAndCompute(frame_gray, mask)
        matches = matcher.knnMatch(desc1, desc0, k=1)
        matching_dict = dict()
        for match in matches:
            if len(match) >=1 and not match[0].trainIdx in matching_dict:
                matching_dict[match[0].trainIdx] = match[0].queryIdx
        for idx, kp0_single in enumerate(kp0):
            if idx in matching_dict:
                x0, y0 = np.int32(kp0_single.pt)
                kp1_single = kp1[matching_dict[idx]]
                x1, y1 = np.int32(kp1_single.pt)
                new_p0.append((x0, y0))
                p1.append((x1, y1))
                st1.append(1)
        p0 = np.array(new_p0)
        p1 = np.array(p1)
        st1 = np.array(st1)
    return p0, p1, st1, img

def calc_transformation(p0, p1, st1):
    idx = np.where(st1 == 1)[0]
    curr = p1[idx, :]
    prev = p0[idx, :]
    [M, inliers] = cv2.estimateAffinePartial2D(np.array(prev), np.array(curr)) # Partial affine to only contains rotation, translation and scaling
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
        plt.subplot(311)
        plt.plot(trajectory_np[:, 0], label="t0_raw")
        plt.plot(trajectory_np_stab[:, 0], label="t0_smoothed")
        plt.ylabel("Pixel")
        plt.legend()
        plt.subplot(312)
        plt.plot(trajectory_np[:, 1], label="t1_raw")
        plt.plot(trajectory_np_stab[:, 1], label="t1_smoothed")
        plt.ylabel("Pixel")
        plt.legend()
        plt.subplot(313)
        plt.plot(trajectory_np[:, 4], label="r_raw")
        plt.plot(trajectory_np_stab[:, 4], label="r_smoothed")
        plt.ylabel("Radian")
        plt.xlabel("t")
        plt.legend()
        plt.savefig("smooth.png")
    diff = trajectory_np_stab - trajectory_np
    transforms_np_stab = transforms_np + diff
    stab_M_list = []
    t0_stab = transforms_np_stab[:, 0]
    t1_stab = transforms_np_stab[:, 1]
    s0_stab = transforms_np[:, 2]
    s1_stab = transforms_np[:, 3] # Use original values for s because it does not change a lot
    r_stab = transforms_np_stab[:, 4] # Use original values for s because it does not change a lot
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
    resized = cv2.resize(frame, (0, 0), fx=args.scale_factor, fy=args.scale_factor)
    new_rows, new_cols, c = resized.shape
    center_x = new_cols //2
    center_y = new_rows //2
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
    prev_frame = None
    extractor = None
    matcher = None
    transforms = []
    count = 0
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame -- or Reached end of video.. Exiting ...")
            break
        logging.debug(count)
        count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.visualize:
            img = frame.copy()
        else:
            img = None

        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        if args.feat_ext != "GFTT" and extractor is None:
            if args.feat_ext == "SIFT":
                extractor = cv2.SIFT_create(nfeatures=args.maxCorners)
                matcher = cv2.BFMatcher(crossCheck=True)
            elif args.feat_ext == "ORB":
                extractor = cv2.ORB_create()
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


        if frame_idx > 0:
            # Feature extraction
            p0 = feature_extract(args, prev_gray, mask, img, extractor=extractor)

            # Optical Flow or Matching
            p0, p1, st1, img = matching(args, prev_gray, frame_gray, p0, mask, img, extractor=extractor, matcher=matcher)

            # Calculate transforms
            M = calc_transformation(p0, p1, st1)
            # if args.debug:
            #     curr_frame = stablize(args, prev_frame, M)
            #     for (x, y), (x0, y0) in zip(p1, p0):
            #         cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            #         cv2.circle(prev_frame, (int(x0), int(y0)), 2, (0, 0, 255), -1)
            #         cv2.polylines(prev_frame, [np.array([np.int32((x0, y0)), np.int32((x, y))]).reshape((-1, 1, 2))], False, (0, 255, 0))
            #         cv2.polylines(frame, [np.array([np.int32((x0, y0)), np.int32((x, y))]).reshape((-1, 1, 2))], False, (0, 255, 0))
            #     plt.imsave("est.png", curr_frame)
            #     plt.imsave("prev.png", prev_frame)
            #     plt.imsave("corr.png", frame)
            transforms = calc_transforms(args, M, transforms)
        
        frame_idx += 1
        prev_gray = frame_gray
        prev_frame = frame
        
        # Show Results
        if img is not None:
            cv2.imshow('Optical Flow', img)

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