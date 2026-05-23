#!/bin/bash

srun -t 100:00:00 \
  -A ealloc_ati-1-neur \
  -J tetris \
  --partition=gpu \
  --gres=gpu:tesla:1 \
  --pty bash