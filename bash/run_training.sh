#!/bin/bash

 run from root folder with: bash bash/run_training.sh
echo "Start experiment"
python steps/training.py \
  experiment=lunar_lander_dqn \
  total_timesteps=10_000 \
  policy_net=custom_mlp
echo "End experiment"


