#!/bin/bash

# run from root folder with: bash bash/run_training.sh
echo "Start experiment"
python steps/training.py \
  experiment=lunar_lander_dqn \
  total_timesteps=20_000
echo "End experiment"
