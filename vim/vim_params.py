# -*- coding:utf-8 -*- 
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# feature_sr = 16000
# feature_frames = 240        # 非独立参数，根据vggish_params算得

# # log mel size (575, 64)
# feature_sr = 44100
# feature_nfft = 2048
# feature_win_len = feature_nfft
# feature_hop_len = 768
# feature_frames = 575        # 非独立参数，根据上面参数算得
# feature_nb_mel_bands = 64
# feature_log_epsilon = 1e-3
# feature_power = 1           # power

# log mel size (240, 64)
feature_sr = 16000
feature_nfft = 1024
feature_win_len = feature_nfft
feature_hop_len = 660
feature_frames = 240        # 非独立参数，根据上面参数算得
feature_nb_mel_bands = 64
feature_log_epsilon = 1e-3
feature_power = 1           # power

# log mel size (240, 64)
# feature_sr = 48000
# feature_nfft = 3072
# feature_win_len = feature_nfft
# feature_hop_len = 1980
# feature_frames = 240        # 非独立参数，根据上面参数算得
# feature_nb_mel_bands = 64
# feature_log_epsilon = 1e-3
# feature_power = 1           # power

# # log mel size (384, 64)
# feature_sr = 16000
# feature_nfft = 1024
# feature_win_len = feature_nfft
# feature_hop_len = 415
# feature_frames = 384        # 非独立参数，根据上面参数算得
# feature_nb_mel_bands = 64
# feature_log_epsilon = 1e-3
# feature_power = 1           # power

# # log mel size (240, 64)
# feature_sr = 16000
# feature_nfft = 1024
# feature_win_len = feature_nfft
# feature_hop_len = 660
# feature_frames = 240        # 非独立参数，根据上面参数算得
# feature_nb_mel_bands = 40
# feature_log_epsilon = 1e-3
# feature_power = 1           # power

# # log mel size (240, 64)
# feature_original_sr = 48000
# feature_sr = 8000
# feature_nfft = 512
# feature_win_len = feature_nfft
# feature_hop_len = 330
# feature_frames = 240        # 非独立参数，根据上面参数算得
# feature_nb_mel_bands = 64
# feature_log_epsilon = 1e-3
# feature_power = 1           # power

BATCH_SIZE = 64
TOTAL_NUM_CLASS = 527
FILE_CLASS_LABELS = '/work/audio/audiosetdl/stat.csv'
DATA_DIR = '/work/audio/flac_path_features'

path_q = 'path-{}'.format(feature_sr)
logmel_q = 'logmel-{}'.format(feature_sr)

