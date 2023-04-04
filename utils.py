import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Video Stablization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training parameters
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
        default=100,
        help="maxCorners for feature extractor"
    )
    parser.add_argument(
        "--qualityLevel",
        type=float,
        default=0.3,
        help="qualityLevel for feature extractor"
    )
    parser.add_argument(
        "--minDistance",
        type=int,
        default=5,
        help="minDistance for feature extractor"
    )
    parser.add_argument(
        "--blockSize",
        type=int,
        default=7,
        help="blockSize for feature extractor"
    )
    parser.add_argument(
        "--trajLen",
        type=int,
        default=100,
        help="Min len of traj"
    )
    parser.add_argument(
        "--DetectInterval",
        type=int,
        default=10,
        help="Update interval"
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
        choices=["GFTT", "SIFT"],
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
        choices=["DeepStab"],
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
        "--degree",
        type=int,
        default=5,
        help="Curve fitting degree"
    )
    args = parser.parse_args()
    return args
    