#!/bin/bash

# run from root folder with: bash bash/run_training.sh
echo "Start experiment"
python steps/training.py \
  experiment=mountain_car_dqn \
  total_timestamps=20_000
echo "End experiment"


#echo "Start experiment"
#python steps/training.py \
#  env=pong_no_frame_skip \
#  model=a2c \
#  total_timestamps=25_000 \
#  policy_net=custom_cnn
#echo "End experiment"