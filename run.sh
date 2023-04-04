#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VS

python optical_flow_stab.py
