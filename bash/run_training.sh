#!/bin/bash

# run from root folder with: bash bash/run_training.sh

echo "Start experiment"
python steps/training.py \
  experiment=mountain_car_a2c \
  total_timesteps=10_000 \
  policy_net=custom_mlp
echo "End experiment"


