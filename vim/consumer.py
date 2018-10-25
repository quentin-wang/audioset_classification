# -*- coding=utf-8 -*-
import pika
import time
import numpy as np
import pickle

credentials = pika.PlainCredentials('myuser', 'mypassword')
with pika.BlockingConnection(
    pika.ConnectionParameters('ice-P910',5672,'myvhost',
    credentials)) as connection:

    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)   # 消息未处理完前不要发送信息的消息

    method_frame, header_frame, body = channel.basic_get(queue='result_queue')
    if method_frame:
        print(method_frame, header_frame)
        aa = pickle.loads(body)
        print(aa)
        channel.basic_ack(method_frame.delivery_tag)
    else:
        print('No message returned')

# $ python consumer.py 
# <Basic.GetOk(['delivery_tag=1', 'exchange=', 'message_count=1', 'redelivered=False', 'routing_key=task_queue'])> <BasicProperties(['delivery_mode=2'])> b'\x80\x03cnumpy.core.multiarray\n_reconstruct\nq\x00cnumpy\nndarray\nq\x01K\x00\x85q\x02C\x01bq\x03\x87q\x04Rq\x05(K\x01K\x02K\x03\x86q\x06cnumpy\ndtype\nq\x07X\x02\x00\x00\x00f8q\x08K\x00K\x01\x87q\tRq\n(K\x03X\x01\x00\x00\x00<q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0cb\x89C0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00q\rtq\x0eb.'

