import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import subprocess
import numpy as np
import multiprocessing as mp

record_cmd = 'arecord --quiet -f S16_LE -r 16000 -c 1 --duration 1 --file-type raw'
# buf_size = 15360    #960 ms data
buf_size = 16000    #960 ms data

def core(q):
    audiobuf = []
    while True:
        p = subprocess.Popen(record_cmd,shell=True,stdout=subprocess.PIPE)
        output,err = p.communicate()
        # print('get {}'.format(len(output)))
        _1sec = np.frombuffer(output, dtype=np.int16)
        _1sec = list(_1sec)
        audiobuf += _1sec

        _960ms = audiobuf[:buf_size]
        audiobuf[0:buf_size] = []
        # print(len(audiobuf))
        q.put(_960ms)

if __name__ == '__main__':
    q = mp.Queue (maxsize=20)
    sys.exit(core(q,) or 0)
