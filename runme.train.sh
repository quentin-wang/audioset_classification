#!/bin/bash
# You need to modify the dataset path. 
# DATA_DIR="/work/audio/audiosetdl"
#DATA_DIR="/work/audio/mfcc_features"
#DATA_DIR="/work/audio/wav_path_features-44.1k"
#DATA_DIR="/work/audio/wav_path_features" obsolete, replaced by vp.DATA_DIR

# You can to modify to your own workspace. 
WORKSPACE=`pwd`/work

BACKEND="keras"     # 'pytorch' | 'keras'

MODEL_TYPE="crnn_sed"
#MODEL_TYPE="decision_level_single_attention"    # 'decision_level_max_pooling'
                                                # | 'decision_level_average_pooling'
                                                # | 'decision_level_single_attention'
                                                # | 'decision_level_multi_attention'

# Train
CUDA_VISIBLE_DEVICES=0,1 python $BACKEND/main.py --workspace=$WORKSPACE --model_type=$MODEL_TYPE train

#CUDA_VISIBLE_DEVICES=1 python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE --mini_data train

#CUDA_VISIBLE_DEVICES=1 python -m cProfile -o profile.stats $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE --mini_data train

# CUDA_VISIBLE_DEVICES=1 python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE train

# Calculate averaged statistics. 
# python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE get_avg_stats
