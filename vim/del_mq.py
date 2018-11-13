# -*- coding=utf-8 -*-
import pika
import sys
import numpy as np
import pickle
import vim_params as vp

credentials = pika.PlainCredentials('myuser', 'mypassword')

with pika.BlockingConnection(
    pika.ConnectionParameters('ice-P910',5672,'myvhost',
    credentials)) as connection:

    # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    q = channel.queue_declare(queue=vp.path_q, durable=True)
    q_len = q.method.message_count
    print('path_q q_len {}'.format(q_len))

    q = channel.queue_declare(queue=vp.logmel_q, durable=True)
    q_len = q.method.message_count
    print('logmel_q q_len {}'.format(q_len))

    channel.queue_delete(queue=vp.path_q)
    channel.queue_delete(queue=vp.logmel_q)
    print('del queue')
