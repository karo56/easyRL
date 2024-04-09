#!/bin/bash

# run from root folder with: bash bash/run_training.sh
#echo "Start experiment"
#python steps/training.py \
#  experiment=lunar_lander_dqn \
#  total_timesteps=20_000 \
#  policy_net=custom_mlp
#echo "End experiment"


echo "Start experiment"
python steps/training.py \
  experiment=pong_with_cnn_ppo \
  total_timesteps=2_000_000
echo "End experiment"