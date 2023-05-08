#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VS

python video_stab.py --save --estModel affine --smooth_win_len 250 --smooth_order 3 --max_frame 250
python video_stab.py --save --estModel homography --smooth_win_len 250 --smooth_order 3 --max_frame 250
