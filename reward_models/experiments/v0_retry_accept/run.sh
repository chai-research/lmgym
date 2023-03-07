#!/bin/bash


export DATA=12000000
python -m torch.distributed.launch --nproc_per_node=4 train.py
