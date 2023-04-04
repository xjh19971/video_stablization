#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VS

python video_stab.py
