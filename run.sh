#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VS

python OpticalFlow.py
