#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VS

# python video_stab.py --video_name 1.avi --save --estModel affine --smooth_win_len 250 --smooth_order 3 --max_frame 250 --debug --feat_ext GFTT
# python video_stab.py --video_name 1.avi --save --estModel homography --smooth_win_len 250 --smooth_order 3 --max_frame 250 --debug --feat_ext GFTT
# python video_stab.py --video_name 3.avi --save --estModel affine --smooth_win_len 250 --smooth_order 3 --max_frame 250 --debug --feat_ext GFTT
# python video_stab.py --video_name 3.avi --save --estModel homography --smooth_win_len 250 --smooth_order 3 --max_frame 250 --debug --feat_ext GFTT
python video_stab.py --video_name 1.mp4 --dataset UAV --save --estModel affine --smooth_win_len 250 --smooth_order 3 --max_frame 1500 --video_offset 900 --debug --feat_ext GFTT
python video_stab.py --video_name 1.mp4 --dataset UAV --save --estModel homography --smooth_win_len 250 --smooth_order 3 --max_frame 1500 --video_offset 900 --debug --feat_ext GFTT