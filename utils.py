import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Video Stablization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training parameters
    parser.add_argument(
        "--estModel",
        type=str,
        default="affine",
        choices=["affine", "homography"],
        help="Estimation model for motion"
    )
    parser.add_argument(
        "--winSize",
        type=int,
        nargs=2,
        default=(15, 15),
        help="Windows size for LKT"
    )
    parser.add_argument(
        "--maxLevel",
        type=int,
        default=2,
        help="Max level for LKT"
    )
    parser.add_argument(
        "--maxCorners",
        type=int,
        default=200,
        help="maxCorners for feature extractor"
    )
    parser.add_argument(
        "--qualityLevel",
        type=float,
        default=0.01,
        help="qualityLevel for feature extractor"
    )
    parser.add_argument(
        "--minDistance",
        type=int,
        default=10,
        help="minDistance for feature extractor"
    )
    parser.add_argument(
        "--blockSize",
        type=int,
        default=20,
        help="blockSize for feature extractor"
    )
    parser.add_argument(
        "--trajLen",
        type=int,
        default=100,
        help="Min len of traj"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize optical flow"
    )
    parser.add_argument(
        "--feat_ext",
        type=str,
        default="GFTT",
        choices=["GFTT", "SIFT", "ORB"],
        help="Feature extractor"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Data folder"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DeepStab",
        choices=["DeepStab", "UAV", "UAV1"],
        help="Dataset folder"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="unstable",
        choices=["unstable", "stable"],
        help="Stable or unstable folder"
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="1.avi",
        help="Video name"
    )
    parser.add_argument(
        "--smooth_win_len",
        type=int,
        default=50,
        help="Window len for smoothing"
    )
    parser.add_argument(
        "--smooth_order",
        type=int,
        default=3,
        help="Order for smoothing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug"
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.2,
        help="Scale factor to remove black boundary"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save videos"
    )
    parser.add_argument(
        "--max_frame",
        type=int,
        default=0,
        help="Maximum number of frames"
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=4,
        default=(105, 0, 590, 1050),
        help="Windows size for UAV video"
    )
    parser.add_argument(
        "--video_offset",
        type=int,
        default=0,
        help="Offset for UAV video"
    )
    args = parser.parse_args()
    return args
    