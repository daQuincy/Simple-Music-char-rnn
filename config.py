#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:41:07 2019

@author: yq
"""

save_path = "experiment_1"

batch_size = 32

rnn = "lstm"
rnn_dim = 512
rnn_layers = 3
dropout = 0.75
learning_rate = 1e-3

min_len = 32
max_len = 128

pitch_depth = 88
max_discrete_times = 32
max_discrete_velocities = 32
max_discrete_durations = 32

infer_len = 200
temperature = 1.0

max_iterations = 20000

resolution = 480
program = 0
