# -*- coding=utf-8 -*-
import sys
import pika
import time
import numpy as np
import pickle
import tasksmq_transfer as tasksmq

# serial
def batch_wav_to_mfcc(x, y, agumentation = False):
    x_mfcc = []
    y_mfcc = []
    len_mfcc = []

    for (ax, ay) in zip(x, y):
        try:
            amfcc, alen = tasksmq.wav_to_mfcc(ax, agumentation)
            x_mfcc.append(amfcc)
            y_mfcc.append(ay)
            len_mfcc.append(alen)
        except Exception as e:
            print(ax)
            print('Error while processing audio: {} '.format(e))

    x_mfcc = np.array(x_mfcc)    # print(x_mfcc.shape)
    y_mfcc = np.array(y_mfcc)
    len_mfcc = np.array(len_mfcc)
    # print(x_mfcc.shape)
    print('.', end='')
    sys.stdout.flush()
    return x_mfcc, y_mfcc, len_mfcc

# def callback(ch, method, properties, body):
#     print(" [x] Received %r" % body)
#     time.sleep(body.count(b'.'))
#     print(" [x] Done")
#     ch.basic_ack(delivery_tag = method.delivery_tag)

def callback(ch, method, properties, body):
    # print(" [x] Received %r" % body)
    x, y = pickle.loads(body)
    x_mfcc, y_mfcc, len_mfcc = batch_wav_to_mfcc(x, y, agumentation = True)
    routing_key = 'result_queue' 
    
    if len(x_mfcc.shape) == 3:
        message = pickle.dumps((x_mfcc,y_mfcc,len_mfcc))   
        channel.basic_publish(exchange='',
                        routing_key=routing_key,
                        body=message,
                        properties=pika.BasicProperties(
                            delivery_mode = 2, # 设置消息为持久化的
                        ))
    ch.basic_ack(delivery_tag = method.delivery_tag)

credentials = pika.PlainCredentials('myuser', 'mypassword')
with pika.BlockingConnection(
    pika.ConnectionParameters('ice-P910',5672,'myvhost',
    credentials)) as connection:

    channel = connection.channel()

    channel.queue_declare(queue='task_queue_transfer', durable=True)  # 设置队列持久化
    channel.queue_declare(queue='result_queue', durable=True) # 设置队列为持久化的队列
    # channel.queue_declare(queue='result_queue2', durable=True) # 设置队列为持久化的队列

    print(' [*] Waiting for messages. To exit press CTRL+C')

    channel.basic_qos(prefetch_count=1)   # 消息未处理完前不要发送信息的消息
    channel.basic_consume(callback,
                        queue='task_queue_transfer')

    channel.start_consuming()

