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
        default=50,
        help="Min len of traj"
    )
    parser.add_argument(
        "--DetectInterval",
        type=int,
        default=10,
        help="Update interval"
    )
    parser.add_argument(
        "--Visualize",
        action="store_true",
        help="Visualize optical flow"
    )
    args = parser.parse_args()
    return args
    