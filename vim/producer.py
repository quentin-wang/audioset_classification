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
    channel.queue_declare(queue='task_queue', durable=True) # 设置队列为持久化的队列

    # Test 1: string
    # message = ' '.join(sys.argv[1:]) or "Hello World!"

    # Test 2: numpy tensor
    message = np.zeros((2,3), dtype=float)
    message = pickle.dumps(message)

    channel.basic_publish(exchange='',
        routing_key='task_queue',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode = 2, # 设置消息为持久化的
                    ))
    # print(" [x] Sent %r" % message)
    # print(message)
