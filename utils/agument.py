#coding=utf-8
import sys, os
import numpy as np
from pydub import AudioSegment

import argparse
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from mfcc import _wavfile
from ctypes import *

libc_so = {"darwin": "libc.dylib", "linux2": "", "linux": ""}[sys.platform]
libc = CDLL(libc_so, use_errno=True, use_last_error=True)

# void* memcpy( void *dest, const void *src, size_t count );
memcpy = libc.memcpy
memcpy.restype = c_void_p
memcpy.argtypes = (c_void_p, c_void_p, c_size_t)    

shmget = libc.shmget
shmget.restype = c_int
shmget.argtypes = (c_int, c_size_t, c_int)

shmat = libc.shmat
shmat.restype = c_void_p
shmat.argtypes = (c_int, c_void_p, c_int)

shmdt = libc.shmdt 
shmdt.restype = c_void_p
shmdt.argtypes = (c_void_p,)

msgget = libc.msgget
msgget.restype = c_int
msgget.argtypes = (c_int, c_int)

msgrcv = libc.msgrcv
msgrcv.restype = c_int
msgrcv.argtypes = (c_int, POINTER(c_byte), c_size_t, c_long, c_int)

msgsnd = libc.msgsnd
msgsnd.restype = c_int
msgsnd.argtypes = (c_int, POINTER(c_byte), c_size_t, c_int)

IPC_CREATE = 512
IPC_EXCL = 0o2000
SHM_KEY = 0x11100       # shm key
SHM_SIZE = 512 * 1024
MSG_SHM_KEY = 0x11100   # msg queue to store shm key.
MSG_SHM_TYPE = 0x01

class ipc_str_msg(Structure):
    _pack_ = 1
    _fields_ = [('type', c_long),('id', c_int)]

class Agumentation(object):
    def __init__(self, shm_cnt):
        self.strategy = {
            'volume': self.volume,
            'wn': self.white_noise,
            'shift': self.shift_sound,
            # 'denoise': self.denoise,
            }
        # self.denoise_init(shm_cnt)

    # Background noise
    def white_noise(self, wav_data):
        NOISE_POWER = 200
        wav_data_int = np.frombuffer(wav_data.raw_data, dtype=np.int16)    # bytes to np.array
        wn = np.random.randint(-NOISE_POWER, NOISE_POWER, size=len(wav_data_int)).astype(np.int16)
        return AudioSegment(data = bytes(wav_data_int + wn),  \
                            sample_width=wav_data.sample_width, frame_rate=wav_data.frame_rate, \
                            channels=wav_data.channels)
    # Volume tuning
    def volume(self, wav_data):
        return wav_data + np.random.randint(-5, 5)

    # Shift tuning
    def shift_sound(self, wav_data):
        wav_data_int = np.frombuffer(wav_data.raw_data, dtype=np.int16)    # bytes to np.array
        roll = int(len(wav_data_int) * np.random.randint(0, 100) * 0.01)     # shift from 0% ~ 99% in random.
        wav_data_int = np.roll(wav_data_int, roll)
        return AudioSegment(data = bytes(wav_data_int),  \
                            sample_width=wav_data.sample_width, frame_rate=wav_data.frame_rate, \
                            channels=wav_data.channels)

    # Speed tuning

    # Denoise
    # def denoise_init(self, shm_cnt):
    #     # assert shm_cnt <= 20, 'shm cnt is too large: {}'.format(shm_cnt)

    #     if (msgget(MSG_SHM_KEY, 0o666 | IPC_EXCL) < 0):    # not exsit, then create
    #         msgid_shm = msgget(MSG_SHM_KEY, 0o666 | IPC_CREATE)
    #         if msgid_shm < 0:
    #             raise Exception("msgid_shm, system not infected")

    #         for ii in range(shm_cnt):
    #             shmid = shmget((SHM_KEY + ii), SHM_SIZE, 0o666 | IPC_CREATE)    # 
    #             if shmid < 0:
    #                 raise Exception("shmget, system not infected")

    #             amsg = ipc_str_msg()
    #             amsg.type = MSG_SHM_TYPE
    #             amsg.id = (SHM_KEY + ii)
    #             print('Denoise send amsg {}'.format(amsg.id))
    #             msgsnd(msgid_shm, cast(byref(amsg), POINTER(c_byte)), sizeof(c_int), MSG_SHM_TYPE, 0)
        
    # def denoise(self, wav_data):
    #     wav_data_int = np.frombuffer(wav_data.raw_data, dtype=np.int16)    # bytes to np.array
    #     amsg = ipc_str_msg()

    #     msgid_shm = msgget(MSG_SHM_KEY, 0o666)
    #     if msgid_shm < 0:
    #         raise Exception("msgid_shm, system not infected")
    #     ret = msgrcv(msgid_shm, cast(byref(amsg), POINTER(c_byte)), sizeof(c_int), MSG_SHM_TYPE, 0)
        
    #     print('get shm {}'.format(amsg.id))
    #     # copy to shm mem
    #     shmid = shmget(amsg.id, SHM_SIZE, 0o666)
    #     if shmid < 0:
    #         raise Exception("shmget, system not infected")

    #     shm_addr = shmat(shmid, None, 0)
    #     wav_len = len(wav_data_int)
    #     memcpy (shm_addr, wav_data_int.ctypes.data, wav_len)
    #     # ret = os.system('./a.out {} {}'.format(amsg.id, wav_len))
    #     memcpy (wav_data_int.ctypes.data, shm_addr, wav_len)

    #     shmdt(shm_addr)
    #     msgsnd(msgid_shm, cast(byref(amsg), POINTER(c_byte)), sizeof(c_int), MSG_SHM_TYPE, 0) # return resource.

    #     return AudioSegment(data = bytes(wav_data_int),  \
    #                         sample_width=2, frame_rate=16000, \
    #                         channels=1)        

##############################################################################

def _run2(data_dir):

    aug = Agumentation(5)

    names = [ na for na in os.listdir(data_dir) if na.endswith('.wav') ]
    for na in names:
        print(na)
        wav_file = data_dir + '/' + na
        wav_data = AudioSegment.from_wav(wav_file)

        wav_data = aug.strategy['denoise'](wav_data)

        raw_wav_data = np.frombuffer(wav_data.raw_data, dtype=np.int16)
        _wavfile.write(wav_file+'.agu.wav', 16000, raw_wav_data)

##############################################################################

AUDIODATA_KEY = 0x300

def _run(data_dir):

    #
    shmid = shmget(AUDIODATA_KEY, SHM_SIZE, 0o666 | IPC_CREATE)    # 
    if shmid < 0:
        raise Exception("shmget, system not infected")
    else:
        shmaddr = shmat(shmid, None, 0)   

    names = [ na for na in os.listdir(data_dir) if na.endswith('.wav') ]
    for na in names:
        print(na)
        wav_file = data_dir + '/' + na
        sr, wav_data = _wavfile.read(wav_file)
        wav_len = len(wav_data) * 2         
        assert wav_len <= SHM_SIZE, 'wav_data is too large: %d' % wav_len
        memcpy (shmaddr, wav_data.ctypes.data, wav_len)
        # ret = os.system('./a.out {} {}'. format(AUDIODATA_KEY, wav_len))
        # if not (ret == 0):
        #     print('a.out process error.')
        #     continue
        memcpy (wav_data.ctypes.data, shmaddr, wav_len)
        _wavfile.write(wav_file+'.agu.wav', 16000, wav_data)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Audio param')
    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where audio data was stored')
    return vars(parser.parse_args())

# Main
if __name__ == '__main__':

    _run2(**parse_arguments())


    
