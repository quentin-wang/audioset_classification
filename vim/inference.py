import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import logging
from sklearn import metrics
from utils import utilities, data_generator, agument

import tensorflow as tf
slim = tf.contrib.slim
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mfcc import vggish_input as mfcc
from mfcc import vggish_params
import vggish_slim

import multiprocessing as mp
# from pydub import AudioSegment

# from tasks import agu_mfcc  # data agumentation

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

import tasksmq

labels = ['Speech', 'Male_speechAND_man_speaking', 'Female_speechAND_woman_speaking'
, 'Child_speechAND_kid_speaking', 'Conversation', 'NarrationAND_monologue'
, 'Babbling', 'Speech_synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell'
, 'Battle_cry', 'Children_shouting', 'Screaming', 'Whispering', 'Laughter'
, 'Baby_laughter', 'Giggle', 'Snicker', 'Belly_laugh', 'ChuckleAND_chortle'
, 'CryingAND_sobbing', 'Baby_cryAND_infant_cry', 'Whimper', 'WailAND_moan'
, 'Sigh', 'Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Male_singing'
, 'Female_singing', 'Child_singing', 'Synthetic_singing', 'Rapping', 'Humming'
, 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 'Gasp', 'Pant'
, 'Snort', 'Cough', 'Throat_clearing', 'Sneeze', 'Sniff', 'Run', 'Shuffle'
, 'WalkAND_footsteps', 'ChewingAND_mastication', 'Biting', 'Gargling'
, 'Stomach_rumble', 'BurpingAND_eructation', 'Hiccup', 'Fart', 'Hands'
, 'Finger_snapping', 'Clapping', 'Heart_soundsAND_heartbeat', 'Heart_murmur'
, 'Cheering', 'Applause', 'Chatter', 'Crowd'
, 'HubbubAND_speech_noiseAND_speech_babble', 'Children_playing', 'Animal'
, 'Domestic_animalsAND_pets', 'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling'
, 'Whimper_(dog)', 'Cat', 'Purr', 'Meow', 'Hiss', 'Caterwaul'
, 'LivestockAND_farm_animalsAND_working_animals', 'Horse', 'Clip-clop'
, 'NeighAND_whinny', 'CattleAND_bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat'
, 'Bleat', 'Sheep', 'Fowl', 'ChickenAND_rooster', 'Cluck'
, 'CrowingAND_cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose'
, 'Honk', 'Wild_animals', 'Roaring_cats_(lionsAND_tigers)', 'Roar', 'Bird'
, 'Bird_vocalizationAND_bird_callAND_bird_song', 'ChirpAND_tweet', 'Squawk'
, 'PigeonAND_dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot'
, 'Bird_flightAND_flapping_wings', 'CanidaeAND_dogsAND_wolves'
, 'RodentsAND_ratsAND_mice', 'Mouse', 'Patter', 'Insect', 'Cricket', 'Mosquito'
, 'FlyAND_housefly', 'Buzz', 'BeeAND_waspAND_etc.', 'Frog', 'Croak', 'Snake'
, 'Rattle', 'Whale_vocalization', 'Music', 'Musical_instrument'
, 'Plucked_string_instrument', 'Guitar', 'Electric_guitar', 'Bass_guitar'
, 'Acoustic_guitar', 'Steel_guitarAND_slide_guitar'
, 'Tapping_(guitar_technique)', 'Strum', 'Banjo', 'Sitar', 'Mandolin', 'Zither'
, 'Ukulele', 'Keyboard_(musical)', 'Piano', 'Electric_piano', 'Organ'
, 'Electronic_organ', 'Hammond_organ', 'Synthesizer', 'Sampler', 'Harpsichord'
, 'Percussion', 'Drum_kit', 'Drum_machine', 'Drum', 'Snare_drum', 'Rimshot'
, 'Drum_roll', 'Bass_drum', 'Timpani', 'Tabla', 'Cymbal', 'Hi-hat', 'Wood_block'
, 'Tambourine', 'Rattle_(instrument)', 'Maraca', 'Gong', 'Tubular_bells'
, 'Mallet_percussion', 'MarimbaAND_xylophone', 'Glockenspiel', 'Vibraphone'
, 'Steelpan', 'Orchestra', 'Brass_instrument', 'French_horn', 'Trumpet'
, 'Trombone', 'Bowed_string_instrument', 'String_section', 'ViolinAND_fiddle'
, 'Pizzicato', 'Cello', 'Double_bass'
, 'Wind_instrumentAND_woodwind_instrument', 'Flute', 'Saxophone', 'Clarinet'
, 'Harp', 'Bell', 'Church_bell', 'Jingle_bell', 'Bicycle_bell', 'Tuning_fork'
, 'Chime', 'Wind_chime', 'Change_ringing_(campanology)', 'Harmonica'
, 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 'Theremin', 'Singing_bowl'
, 'Scratching_(performance_technique)', 'Pop_music', 'Hip_hop_music'
, 'Beatboxing', 'Rock_music', 'Heavy_metal', 'Punk_rock', 'Grunge'
, 'Progressive_rock', 'Rock_and_roll', 'Psychedelic_rock', 'Rhythm_and_blues'
, 'Soul_music', 'Reggae', 'Country', 'Swing_music', 'Bluegrass', 'Funk'
, 'Folk_music', 'Middle_Eastern_music', 'Jazz', 'Disco', 'Classical_music'
, 'Opera', 'Electronic_music', 'House_music', 'Techno', 'Dubstep'
, 'Drum_and_bass', 'Electronica', 'Electronic_dance_music', 'Ambient_music'
, 'Trance_music', 'Music_of_Latin_America', 'Salsa_music', 'Flamenco', 'Blues'
, 'Music_for_children', 'New-age_music', 'Vocal_music', 'A_capella'
, 'Music_of_Africa', 'Afrobeat', 'Christian_music', 'Gospel_music'
, 'Music_of_Asia', 'Carnatic_music', 'Music_of_Bollywood', 'Ska'
, 'Traditional_music', 'Independent_music', 'Song', 'Background_music'
, 'Theme_music', 'Jingle_(music)', 'Soundtrack_music', 'Lullaby'
, 'Video_game_music', 'Christmas_music', 'Dance_music', 'Wedding_music'
, 'Happy_music', 'Funny_music', 'Sad_music', 'Tender_music', 'Exciting_music'
, 'Angry_music', 'Scary_music', 'Wind', 'Rustling_leaves'
, 'Wind_noise_(microphone)', 'Thunderstorm', 'Thunder', 'Water', 'Rain'
, 'Raindrop', 'Rain_on_surface', 'Stream', 'Waterfall', 'Ocean', 'WavesAND_surf'
, 'Steam', 'Gurgling', 'Fire', 'Crackle', 'Vehicle', 'BoatAND_Water_vehicle'
, 'SailboatAND_sailing_ship', 'RowboatAND_canoeAND_kayak'
, 'MotorboatAND_speedboat', 'Ship', 'Motor_vehicle_(road)', 'Car'
, 'Vehicle_hornAND_car_hornAND_honking', 'Toot', 'Car_alarm'
, 'Power_windowsAND_electric_windows', 'Skidding', 'Tire_squeal'
, 'Car_passing_by', 'Race_carAND_auto_racing', 'Truck', 'Air_brake'
, 'Air_hornAND_truck_horn', 'Reversing_beeps'
, 'Ice_cream_truckAND_ice_cream_van', 'Bus', 'Emergency_vehicle'
, 'Police_car_(siren)', 'Ambulance_(siren)'
, 'Fire_engineAND_fire_truck_(siren)', 'Motorcycle'
, 'Traffic_noiseAND_roadway_noise', 'Rail_transport', 'Train', 'Train_whistle'
, 'Train_horn', 'Railroad_carAND_train_wagon', 'Train_wheels_squealing'
, 'SubwayAND_metroAND_underground', 'Aircraft', 'Aircraft_engine'
, 'Jet_engine', 'PropellerAND_airscrew', 'Helicopter'
, 'Fixed-wing_aircraftAND_airplane', 'Bicycle', 'Skateboard', 'Engine'
, 'Light_engine_(high_frequency)', "Dental_drillAND_dentist's_drill"
, 'Lawn_mower', 'Chainsaw', 'Medium_engine_(mid_frequency)'
, 'Heavy_engine_(low_frequency)', 'Engine_knocking', 'Engine_starting'
, 'Idling', 'AcceleratingAND_revvingAND_vroom', 'Door', 'Doorbell', 'Ding-dong'
, 'Sliding_door', 'Slam', 'Knock', 'Tap', 'Squeak', 'Cupboard_open_or_close'
, 'Drawer_open_or_close', 'DishesAND_potsAND_and_pans'
, 'CutleryAND_silverware', 'Chopping_(food)', 'Frying_(food)'
, 'Microwave_oven', 'Blender', 'Water_tapAND_faucet'
, 'Sink_(filling_or_washing)', 'Bathtub_(filling_or_washing)', 'Hair_dryer'
, 'Toilet_flush', 'Toothbrush', 'Electric_toothbrush', 'Vacuum_cleaner'
, 'Zipper_(clothing)', 'Keys_jangling', 'Coin_(dropping)', 'Scissors'
, 'Electric_shaverAND_electric_razor', 'Shuffling_cards', 'Typing'
, 'Typewriter', 'Computer_keyboard', 'Writing', 'Alarm', 'Telephone'
, 'Telephone_bell_ringing', 'Ringtone', 'Telephone_dialingAND_DTMF'
, 'Dial_tone', 'Busy_signal', 'Alarm_clock', 'Siren', 'Civil_defense_siren'
, 'Buzzer', 'Smoke_detectorAND_smoke_alarm', 'Fire_alarm', 'Foghorn', 'Whistle'
, 'Steam_whistle', 'Mechanisms', 'RatchetAND_pawl', 'Clock', 'Tick', 'Tick-tock'
, 'Gears', 'Pulleys', 'Sewing_machine', 'Mechanical_fan', 'Air_conditioning'
, 'Cash_register', 'Printer', 'Camera', 'Single-lens_reflex_camera', 'Tools'
, 'Hammer', 'Jackhammer', 'Sawing', 'Filing_(rasp)', 'Sanding', 'Power_tool'
, 'Drill', 'Explosion', 'GunshotAND_gunfire', 'Machine_gun', 'Fusillade'
, 'Artillery_fire', 'Cap_gun', 'Fireworks', 'Firecracker', 'BurstAND_pop'
, 'Eruption', 'Boom', 'Wood', 'Chop', 'Splinter', 'Crack', 'Glass'
, 'ChinkAND_clink', 'Shatter', 'Liquid', 'SplashAND_splatter', 'Slosh', 'Squish'
, 'Drip', 'Pour', 'TrickleAND_dribble', 'Gush', 'Fill_(with_liquid)', 'Spray'
, 'Pump_(liquid)', 'Stir', 'Boiling', 'Sonar', 'Arrow'
, 'WhooshAND_swooshAND_swish', 'ThumpAND_thud', 'Thunk', 'Electronic_tuner'
, 'Effects_unit', 'Chorus_effect', 'Basketball_bounce', 'Bang', 'SlapAND_smack'
, 'WhackAND_thwack', 'SmashAND_crash', 'Breaking', 'Bouncing', 'Whip', 'Flap'
, 'Scratch', 'Scrape', 'Rub', 'Roll', 'Crushing', 'CrumplingAND_crinkling'
, 'Tearing', 'BeepAND_bleep', 'Ping', 'Ding', 'Clang', 'Squeal', 'Creak', 'Rustle'
, 'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Clickety-clack', 'Rumble', 'Plop'
, 'JingleAND_tinkle', 'Hum', 'Zing', 'Boing', 'Crunch', 'Silence', 'Sine_wave'
, 'Harmonic', 'Chirp_tone', 'Sound_effect', 'Pulse', 'InsideAND_small_room'
, 'InsideAND_large_room_or_hall', 'InsideAND_public_space'
, 'OutsideAND_urban_or_manmade', 'OutsideAND_rural_or_natural'
, 'Reverberation', 'Echo', 'Noise', 'Environmental_noise', 'Static', 'Mains_hum'
, 'Distortion', 'Sidetone', 'Cacophony', 'White_noise', 'Pink_noise'
, 'Throbbing', 'Vibration', 'Television', 'Radio', 'Field_recording']

_TOTAL_NUM_CLASS = 527
__FILE_CLASS_LABELS = '../audiosetdl/class_labels_indices-train.csv'
# _CHECK_POINT = 'work/models/main/balance_type=balance_in_batch/model_type=decision_level_average_pooling/checkpoint'
MODEL_PATH = 'work/models/main/balance_type=balance_in_batch/model_type=decision_level_average_pooling/'
meta_path = MODEL_PATH + 'md_{}_iters.ckpt.meta'.format(29000)
model_path = MODEL_PATH + 'md_{}_iters.ckpt'.format(29000)

def print_prediction(wavname, out, mfcclen):
    name = str(wavname,'utf-8')
    print(name)
    print('mfcclen', mfcclen)
    sorted_out = np.argsort(-out)
    # print('out[:100]')
    # print(out[:100])
    # print(out[sorted_out[:100]])
    
    for i in range(3):
        print('{} \t prob: {}'.format(labels[sorted_out[i]], out[sorted_out[i]]))

def get_name_list(audio_path):
    result = []
    flist = os.listdir(audio_path)
    for afile in flist:
        if os.path.basename(afile).split('.')[-1] == 'wav':     # name.wav.short bug fix
            afile_path = bytes(os.path.join(audio_path, afile), encoding='utf-8')
            result.append(afile_path)

    return result

def core(args):

    data_dir = args.data_dir
    workspace = args.workspace
    filename = args.filename
    model_type = args.model_type
    # model = args.model

    # Load data
    load_time = time.time()

    # compute test directory files' MFCC
    test_x = get_name_list(data_dir)
    test_y = []
    test_x_mfcc, _, seq_len = tasksmq.batch_wav_to_mfcc_parallel(test_x, test_y, agumentation=False)

    # tensorflow graph
    saver = tf.train.import_meta_graph(meta_path)   # import graph

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        # graph = tf.get_default_graph()

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        sequence_length_tensor = sess.graph.get_tensor_by_name(
            'vggish/input_sequence_length:0')
        output_tensor = sess.graph.get_tensor_by_name(
            'mix/prediction:0')
        # all_output_tensor = sess.graph.get_tensor_by_name(
        #     'vggish/Reshape_2:0')
        # [output, all_outputs] = sess.run([output_tensor, all_output_tensor], feed_dict={features_tensor: test_x_mfcc, sequence_length: [120]})       #output = model.predict(input)
        [output, se] = sess.run([output_tensor, sequence_length_tensor], feed_dict={features_tensor: test_x_mfcc, sequence_length_tensor: [80]})       #output = model.predict(input)
        
        # for (ii,xx) in enumerate(all_outputs):
        #     print(ii)
        #     print(xx[:10])

        print(se)

        # iter all files
        for ii in range(len(test_x)):
            print_prediction(test_x[ii], output[ii], seq_len[ii])


    # train_x = []
    # train_y = []
    # train_id_list = []
    # test_x = []
    # test_y = []
    # test_id_list = []

    # for aclass in labels_dict['name']:
    #     print(aclass)

    #     local_train_x = []
    #     local_train_y = []
    #     local_train_id_list = []
    #     local_test_x = []
    #     local_test_y = []
    #     local_test_id_list = []

    #     # Path of hdf5 data
    #     bal_train_hdf5_path = os.path.join(data_dir, aclass, "balanced_train_segments.hdf5")
    #     unbal_train_hdf5_path = os.path.join(data_dir, aclass, "unbalanced_train_segments.hdf5")
    #     test_hdf5_path = os.path.join(data_dir, aclass, "eval_segments.hdf5")

    #     if mini_data:
    #         # Only load balanced data
    #         (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
    #             bal_train_hdf5_path)

    #         local_train_x = bal_train_x
    #         local_train_y = bal_train_y
    #         local_train_id_list = bal_train_id_list

    #     else:
    #         # Load both balanced and unbalanced data
    #         (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
    #             bal_train_hdf5_path)

    #         (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
    #             unbal_train_hdf5_path)

    #         local_train_x = np.concatenate((bal_train_x, unbal_train_x))
    #         local_train_y = np.concatenate((bal_train_y, unbal_train_y))
    #         local_train_id_list = bal_train_id_list + unbal_train_id_list

    #     labels_dict['count'].append(len(local_train_id_list))
    #     # Test data
    #     (local_test_x, local_test_y, local_test_id_list) = utilities.load_data(test_hdf5_path)

    #     train_x = ( local_train_x if (train_x == []) else np.concatenate((train_x, local_train_x)) )
    #     train_y = ( local_train_y if (train_y == []) else np.concatenate((train_y, local_train_y)) )
    #     train_id_list = train_id_list + local_train_id_list
    #     test_x = ( local_test_x if (test_x == []) else np.concatenate((test_x, local_test_x)) )
    #     test_y = ( local_test_y if (test_y == []) else np.concatenate((test_y, local_test_y)) )
    #     test_id_list = test_id_list + local_test_id_list

    # # Mask other classes.
    # for ii, item in  enumerate(train_y):
    #     train_y[ii] = np.logical_and(item, labels_map_mask)
    # for ii, item in  enumerate(test_y):
    #     test_y[ii] = np.logical_and(item, labels_map_mask)

    # for ii, item in  enumerate(test_y):
    #     if not any(item):
    #         print(test_id_list[ii])
    #         print(ii, item)
    #         raise Exception('False item, no positive label.')

    # test_x_mfcc, test_y_mfcc = tasksmq.batch_wav_to_mfcc_parallel(test_x, test_y, agumentation=False)
    # test_seq_len = np.ones(len(test_x_mfcc)) * 240     # length array of the batch

    # logging.info("Loading data time: {:.3f} s".format(time.time() - load_time))
    # logging.info("Training data shape: {}".format(train_x.shape))

    # # Output directories
    # sub_dir = os.path.join(filename,
    #                        'balance_type={}'.format(balance_type),
    #                        'model_type={}'.format(model_type))

    # models_dir = os.path.join(workspace, "models", sub_dir)
    # utilities.create_folder(models_dir)

    # stats_dir = os.path.join(workspace, "stats", sub_dir)
    # utilities.create_folder(stats_dir)

    # probs_dir = os.path.join(workspace, "probs", sub_dir)
    # utilities.create_folder(probs_dir)

    # # Data generator
    # if balance_type == 'no_balance':
    #     DataGenerator = data_generator.VanillaDataGenerator

    # elif balance_type == 'balance_in_batch':
    #     DataGenerator = data_generator.BalancedDataGenerator

    # else:
    #     raise Exception("Incorrect balance_type!")

    # train_gen = DataGenerator(
    #     x=train_x,
    #     y=train_y,
    #     batch_size=batch_size,
    #     labels_map=labels_dict['id'],
    #     shuffle=True,
    #     seed=1234)

    # # create work thread for DataGenerator
    # # q_batch = mp.Queue (maxsize=10)
    # task_generate_batch = mp.Process (target = tasksmq.generate_batch, args = (train_gen,))
    # task_generate_batch.start()


    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # # with tf.Graph().as_default(), tf.Session() as sess:
    #     embeddings = vggish_slim.define_vggish_slim(training=True)     # training=False，模型参数不可被修改
        
    #     with tf.variable_scope('mix'):
    #         # pretend to have _TOTAL_NUM_CLASS class, but feed _NUM_CLASS actually.
    #         # Add a fully connected layer
    #         # fc_tensor = slim.fully_connected(embeddings, 4096, activation_fn=tf.nn.relu, scope='fc1')
    #         # logits_tensor = slim.fully_connected(embeddings, _TOTAL_NUM_CLASS, activation_fn=None, scope='logits')
    #         logits_tensor = embeddings
    #         output_tensor = tf.sigmoid(embeddings, name='prediction')

    #         # Add training ops.
    #         with tf.variable_scope('train'):
    #             global_step = tf.Variable(
    #                 0, name='global_step', trainable=False,
    #                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
    #                             tf.GraphKeys.GLOBAL_STEP])

    #             labels = tf.placeholder(
    #                 tf.float32, shape=(None, _TOTAL_NUM_CLASS), name='labels')

    #             xent = tf.nn.sigmoid_cross_entropy_with_logits(
    #                 logits=logits_tensor, labels=labels, name='xent')
    #             loss = tf.reduce_mean(xent, name='loss_op')
                
    #             learning_rate = tf.train.exponential_decay(
    #                 init_learning_rate, global_step=global_step, decay_steps=50000, decay_rate=1e-6)

    #             optimizer = tf.train.AdamOptimizer(
    #                 learning_rate=learning_rate,    # vggish_params.LEARNING_RATE,
    #                 epsilon=vggish_params.ADAM_EPSILON)
    #             optimizer.minimize(loss, global_step=global_step, name='train_op')   

    #     # Initialize all variables in the model, and then load the pre-trained
    #     # VGGish checkpoint.
    #     sess.run(tf.global_variables_initializer())
    #     # vggish_slim.load_vggish_slim_checkpoint(sess, _CHECK_POINT)
    #     saver = tf.train.Saver(max_to_keep=50)

    #     # Locate all the tensors and ops we need for the training loop.
    #     features_tensor = sess.graph.get_tensor_by_name(
    #         vggish_params.INPUT_TENSOR_NAME)
    #     sequence_length = sess.graph.get_tensor_by_name(
    #         'vggish/input_sequence_length:0')

    #     labels_tensor = sess.graph.get_tensor_by_name('mix/train/labels:0')
    #     global_step_tensor = sess.graph.get_tensor_by_name('mix/train/global_step:0')
    #     loss_tensor = sess.graph.get_tensor_by_name('mix/train/loss_op:0')
    #     train_op = sess.graph.get_operation_by_name('mix/train/train_op')

    #     if quantize:
    #         # Call the training rewrite quantize
    #         tf.contrib.quantize.create_training_graph(quant_delay=100)

    #     #----- tensorboard-------
    #     tf.summary.scalar('loss', loss_tensor)
    #     tf.summary.scalar('step', global_step_tensor)
    #     tf.summary.histogram('output_act', output_tensor)

    #     merged = tf.summary.merge_all()
    #     train_writer = tf.summary.FileWriter('tblogs',sess.graph)

    #     train_time = time.time()

    #     credentials = pika.PlainCredentials('myuser', 'mypassword')
    #     with pika.BlockingConnection(
    #         pika.ConnectionParameters('ice-P910',5672,'myvhost',
    #         credentials)) as connection:

    #         channel = connection.channel()
    #         channel.basic_qos(prefetch_count=1)   # 消息未处理完前不要发送信息的消息

    #         while True:
    #             method_frame, header_frame, body = channel.basic_get(queue='result_queue')      # consumer
    #             if method_frame:
    #                 print(method_frame, header_frame)
    #                 batch_x_mfcc, batch_y_mfcc = cPickle.loads(body)
    #                 # print(batch_x_mfcc.shape)      # print(batch_y_mfcc.shape)
    #                 channel.basic_ack(method_frame.delivery_tag)

    #                 if method_frame.message_count > 400:         # consumer can't process too much messages.
    #                     for _ in range(100):
    #                         method_frame, header_frame, body = channel.basic_get(queue='result_queue')
    #                         channel.basic_ack(method_frame.delivery_tag)

    #             else:
    #                 # print('No message returned')
    #                 time.sleep(0.5)
    #                 continue

    #             [num_steps, loss, _] = sess.run(
    #                         [global_step_tensor, loss_tensor, train_op],
    #                         feed_dict={features_tensor: batch_x_mfcc, labels_tensor: batch_y_mfcc, sequence_length:batch_seq_len})
    #             print(num_steps,':', loss)

    #             summary = sess.run(merged, feed_dict={features_tensor: batch_x_mfcc, labels_tensor: batch_y_mfcc, sequence_length:batch_seq_len})
    #             train_writer.add_summary(summary, num_steps)
                
    #             # with tf.name_scope('GPU_%d' % 1) as scope:

    #             if num_steps % 1000 == 0:
    #                 logging.info("------------------")
    #                 logging.info(
    #                     "Iteration: {}, train time: {:.3f} s".format(
    #                         num_steps, time.time() - train_time))

    #                 logging.info("Test statistics:")

    #                 # tensorflow/core/framework/allocator.cc:101] Allocation of 561807360 exceeds 10% of system memory.
    #                 output = []
    #                 test_len = len(test_x_mfcc)
    #                 start_pos = 0
    #                 max_iter_len = batch_size   # 300 overflow GPU
    #                 while True:
    #                     print(start_pos)
    #                     iter_test_len = ((test_len - start_pos) if ((test_len - start_pos) < max_iter_len) else max_iter_len)
    #                     if (iter_test_len <= 0):
    #                         break
    #                     local_output = sess.run(output_tensor, feed_dict={features_tensor: test_x_mfcc[start_pos:start_pos+iter_test_len], 
    #                         sequence_length: test_seq_len[start_pos:start_pos+iter_test_len]})
    #                     print('local_output.shape', local_output.shape)
    #                     output = ( local_output if (output == []) else np.concatenate((output, local_output)) )
    #                     start_pos += iter_test_len
                    
    #                 # output = sess.run(output_tensor, feed_dict={features_tensor: test_x_mfcc})       #output = model.predict(input)
    #                 print('output', output.shape)
    #                 print('test_y_mfcc', test_y_mfcc.shape)

    #                 mAP, _, AP = tf_evaluate(
    #                     target=test_y_mfcc,
    #                     output=output,
    #                     stats_dir=os.path.join(stats_dir, "test"),
    #                     probs_dir=os.path.join(probs_dir, "test"),
    #                     iteration=num_steps,
    #                     labels_map=labels_dict['id'])
    #                 labels_dict['AP'] = AP
    #                 for (name, ap) in zip(labels_dict['name'], labels_dict['AP']):
    #                     print(name, '\t', ap)
    #                 infer = np.argsort (-output[0])
    #                 print(output[0][infer][:40])
    #                 train_time = time.time()
    #                 rplot(labels_dict['name'], labels_dict['count'], labels_dict['AP'])

    #             # Save model
    #             if (num_steps % 500) == 0:
    #                 save_out_path = os.path.join(
    #                     models_dir, "md_{}_iters.ckpt".format(num_steps))
    #                 save_path = saver.save(sess, save_out_path)

    #                 # Save the checkpoint and eval graph proto to disk for freezing
    #                 # and providing to TFLite.
    #                 # eval_graph_file = os.path.join(
    #                 #     models_dir, "md_{}_iters.pb".format(num_steps))
    #                 # with open(eval_graph_file, 'w') as f:
    #                 #     f.write(str(tf.get_default_graph().as_graph_def()))
                
    #             # Stop training when maximum iteration achieves
    #             if num_steps == 500001:
    #                 break

    # task_generate_batch.terminate()
    # train_writer.close()
