# -*- coding=utf-8 -*-
import pika
import sys
import numpy as np
import pickle

credentials = pika.PlainCredentials('myuser', 'mypassword')

with pika.BlockingConnection(
    pika.ConnectionParameters('ice-P910',5672,'myvhost',
    credentials)) as connection:

    # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    q = channel.queue_declare(queue='task_queue_transfer', durable=True)
    q_len = q.method.message_count
    print('task_queue q_len {}'.format(q_len))

    q = channel.queue_declare(queue='result_queue', durable=True)
    q_len = q.method.message_count
    print('result_queue q_len {}'.format(q_len))

    channel.queue_delete(queue='task_queue_transfer')
    channel.queue_delete(queue='result_queue')
    print('del task_queue & result_queue')
