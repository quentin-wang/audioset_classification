#!/bin/sh

mpirun -np 1  -H localhost:1 -bind-to none -map-by slot     -x NCCL_DEBUG=INFO -mca pml ob1 -mca btl ^openib     python vim/main.py --data_dir="/work/audio/wav_path_features" --workspace="work" --model_type="decision_level_average_pooling" transfer

