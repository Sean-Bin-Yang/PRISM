#!/usr/bin/env bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8


python MoE_train.py \
  --city NY --task serviceCall --seed 42 --fusion_mode router \
  --auto_weight \
  --use_pid_mse --pid_alpha 1.0 --pid_beta 0.04 --pid_gamma 0.01 --pid_leak 0.99 --pid_dim 1 --router_temp 2 --verbose_flops
