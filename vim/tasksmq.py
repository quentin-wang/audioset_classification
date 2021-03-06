# -*- coding:utf-8 -*-
import sys, os
import numpy as np
# from pydub import AudioSegment
import librosa
import soundfile
import multiprocessing as mp
import re

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from mfcc import vggish_input as mfcc
# from utils import agument
import time
import pika

from utils import utilities, data_generator

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

import pandas as pd
import vim_params as vp

# aug = agument.Agumentation(mp.cpu_count())

def generate_batch(train_gen, ):
    print('run generate_batch ...')

    credentials = pika.PlainCredentials('myuser', 'mypassword')

    with pika.BlockingConnection(
        pika.ConnectionParameters('ice-P910',5672,'myvhost',
        credentials)) as connection:

        # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        print('channel = connection.channel()')

        for (batch_x, batch_y) in train_gen.generate():
            q = channel.queue_declare(queue=vp.path_q, durable=True) # 设置队列为持久化的队列
            message = cPickle.dumps((batch_x, batch_y))
            channel.basic_publish(exchange='',
                routing_key=vp.path_q,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode = 2, # 设置消息为持久化的
                            ))
            time.sleep(0.2)
            q_len = q.method.message_count

            if q_len > 20:
                print('path_q runs too fast, sleep.............................')
                time.sleep(5)
            
def start_process():
    pass
    # print('Starting',mp.current_process().name)

def batch_wav_to_mfcc_parallel(x, y, agumentation = False):
    x_mfcc = []
    x_len = []

    pool = mp.Pool(processes=None,initializer=start_process,)
    results = [pool.apply_async(wav_to_mfcc, (t,agumentation,)) for t in x]         # results = pool.imap(wav_to_mfcc, x, mp.cpu_count())      # imap(func, iterable[, chunksize]) # map_async

    pool.close()
    pool.join()

    for aresult in results:
        amfcc, alen = aresult.get()
        x_mfcc.append(amfcc)
        x_len.append(alen)

    x_mfcc = np.array(x_mfcc)
    x_len = np.array(x_len)
    
    print('x_mfcc', x_mfcc.shape)
    return x_mfcc, y, x_len

strinfo = re.compile('downloads_merge')
def read_audio(path, target_fs=None):
    if 'extra_sound' in path:      # extra audio
        print(path)
        (audio, fs) = soundfile.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if target_fs is not None and fs != target_fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
            fs = target_fs
    else:
        save_path = strinfo.sub('downloads-flac-{}'.format(target_fs), path)
        if os.path.exists(save_path):       # after converted
            (audio, fs) = soundfile.read(save_path)
        else:   # need to be converted
            print(save_path, 'is not exist!!!')
            (audio, fs) = soundfile.read(path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if target_fs is not None and fs != target_fs:
                audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
                fs = target_fs
            # save 
            try:
                soundfile.write(save_path, audio, fs)
            except Exception as e:
                print(e)
                newpath = os.path.dirname(save_path)
                if not os.path.exists(newpath):
                    print(newpath, "is not exist. ")
                    os.makedirs(newpath, mode=0o755)
                soundfile.write(save_path, audio, fs)
    return audio, fs

def wav_to_mfcc(ax, agumentation=False):

    # sr, wav_data = _wavfile.read(str(ax,'utf-8'))
    # assert sr == 16000, 'Bad sample: {}'.format(sr)
    axs = str(ax,'utf-8')
    (audio, sr) = read_audio(axs, target_fs=vp.feature_sr)      # wav and flac
    
    if agumentation:
        # 50 % to execute white noise
        if (np.random.randint(0, 2) == 0):
            NOISE_POWER = 200
            wn = (np.random.randint(-NOISE_POWER, NOISE_POWER, size=len(audio))) / 32768
            audio += wn
    
        # 50 % to execute volume
        
        # 50 % to execute shift
        if (np.random.randint(0, 2) == 0):
            roll = int(len(audio) * np.random.randint(0, 1000) * 0.001)     # shift from 0% ~ 99% in random.
            audio = np.roll(audio, roll)

    try:
        amfcc = np.log(librosa.feature.melspectrogram(audio,  # / 32768.0
            sr=vp.feature_sr, n_mels=vp.feature_nb_mel_bands, 
            n_fft=vp.feature_nfft, hop_length=vp.feature_hop_len, 
            power=vp.feature_power) + vp.feature_log_epsilon)
        amfcc = amfcc.T
        # print('amfcc.shape ', amfcc.shape)

        alen = amfcc.shape[0]
        if (alen < vp.feature_frames):
            amfcc = np.concatenate((amfcc, np.zeros(shape=((vp.feature_frames - alen), amfcc.shape[1]))), axis=0)
            
        elif (alen > vp.feature_frames):
            alen = vp.feature_frames
            amfcc = amfcc[:vp.feature_frames]

    except Exception as e:
        print(axs)
        print('Error while processing audio: {} '.format(e))

    return amfcc, alen


# def wav_to_mfcc(ax, agumentation=False):

#     # sr, wav_data = _wavfile.read(str(ax,'utf-8'))
#     # assert sr == 16000, 'Bad sample: {}'.format(sr)
#     axs = str(ax,'utf-8')
#     wav_data = AudioSegment.from_wav(axs)

#     if agumentation:
#         # 50 % to execute white noise
#         if (np.random.randint(0, 2) == 0):
#             wav_data = aug.strategy['wn'](wav_data)

#         # 50 % to execute volume
#         # if (np.random.randint(0, 2) == 0):
#         #     wav_data = aug.strategy['volume'](wav_data)

#         # 50 % to execute shift
#         if (np.random.randint(0, 2) == 0):
#             wav_data = aug.strategy['shift'](wav_data)

#     raw_wav_data = np.frombuffer(wav_data.raw_data, dtype=np.int16)

#     try:
#         assert wav_data.frame_rate == vp.feature_sr, 'Bad sample type: %r' % wav_data.frame_rate

#         # if vp.feature_sr == 16000:
#         #     amfcc = mfcc.waveform_to_examples(raw_wav_data / 32768.0, wav_data.frame_rate)
#         # elif vp.feature_sr == 44100:
#         amfcc = np.log(librosa.feature.melspectrogram(raw_wav_data / 32768.0, 
#             sr=vp.feature_sr, n_mels=vp.feature_nb_mel_bands, 
#             n_fft=vp.feature_nfft, hop_length=vp.feature_hop_len, 
#             power=vp.feature_power) + vp.feature_log_epsilon)
#         amfcc = amfcc.T
#         # print('amfcc.shape ', amfcc.shape)

#         alen = amfcc.shape[0]
#         if (alen < vp.feature_frames):
#             amfcc = np.concatenate((amfcc, np.zeros(shape=((vp.feature_frames - alen), amfcc.shape[1]))), axis=0)
            
#         elif (alen > vp.feature_frames):
#             alen = vp.feature_frames
#             amfcc = amfcc[:vp.feature_frames]

#     except Exception as e:
#         print(axs)
#         print('Error while processing audio: {} '.format(e))

#     return amfcc, alen


if __name__ == '__main__':
    data_dir = vp.DATA_DIR
    mini_data = False

    df = pd.read_csv(vp.FILE_CLASS_LABELS)
    labels_dict = {}
    labels_dict['name'] = np.array(df[df['train100'] == 1]['display_name'])
    labels_dict['id'] = np.array(df[df['train100'] == 1]['index'])
    labels_dict['count'] = []

    train_x = []
    train_y = []
    train_id_list = []

    labels_map_mask = [False] * vp.TOTAL_NUM_CLASS
    for x in labels_dict['id']:
        labels_map_mask[x] = True

    # Load data
    load_time = time.time()

    train_x = []
    train_y = []
    train_id_list = []

    for aclass in labels_dict['name']:
        print(aclass)

        local_train_x = []
        local_train_y = []
        local_train_id_list = []

        # Path of hdf5 data
        bal_train_hdf5_path = os.path.join(data_dir, aclass, "balanced_train_segments.hdf5")
        unbal_train_hdf5_path = os.path.join(data_dir, aclass, "unbalanced_train_segments.hdf5")
        test_hdf5_path = os.path.join(data_dir, aclass, "eval_segments.hdf5")

        if mini_data:
            # Only load balanced data
            (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
                bal_train_hdf5_path)

            local_train_x = bal_train_x
            local_train_y = bal_train_y
            local_train_id_list = bal_train_id_list

        else:
            # Load both balanced and unbalanced data
            (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
                bal_train_hdf5_path)

            (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
                unbal_train_hdf5_path)

            local_train_x = np.concatenate((bal_train_x, unbal_train_x))
            local_train_y = np.concatenate((bal_train_y, unbal_train_y))
            local_train_id_list = bal_train_id_list + unbal_train_id_list

        labels_dict['count'].append(len(local_train_id_list))

        train_x = ( local_train_x if (train_x == []) else np.concatenate((train_x, local_train_x)) )
        train_y = ( local_train_y if (train_y == []) else np.concatenate((train_y, local_train_y)) )
        train_id_list = train_id_list + local_train_id_list

    # Mask other classes.
    for ii, item in  enumerate(train_y):
        train_y[ii] = np.logical_and(item, labels_map_mask)

    DataGenerator = data_generator.BalancedDataGenerator

    train_gen = DataGenerator(
        x=train_x,
        y=train_y,
        batch_size=vp.BATCH_SIZE,
        labels_map=labels_dict['id'],
        shuffle=True,
        seed=1234)

    generate_batch(train_gen)
 
