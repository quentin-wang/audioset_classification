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

import multiprocessing as mp
import audio_sample

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
# MODEL_PATH = 'work/models/main/balance_type=balance_in_batch/model_type=decision_level_average_pooling/'
MODEL_PATH = 'checkpoints/'
meta_path = MODEL_PATH + 'model.ckpt-{}.meta'.format(10000)
model_path = MODEL_PATH + 'model.ckpt-{}'.format(10000)

_DURATION = 3   # seconds

def print_prediction(out, mfcclen):
    # print('mfcclen', mfcclen)
    sorted_out = np.argsort(-out)
    # print(out[sorted_out[:100]])

    for i in range(3):
        if out[sorted_out[i]] > 0.3:
            print('{} \t prob: {}'.format(labels[sorted_out[i]], out[sorted_out[i]]))

    print('')

def get_name_list(audio_path):
    result = []
    flist = os.listdir(audio_path)
    for afile in flist:
        if os.path.basename(afile).split('.')[-1] == 'wav':     # name.wav.short bug fix
            afile_path = bytes(os.path.join(audio_path, afile), encoding='utf-8')
            result.append(afile_path)

    return result


def raw2mfcc(raw_wav_data):
    raw_wav_data = np.array(raw_wav_data)
    try:
        amfcc = mfcc.waveform_to_examples(raw_wav_data / 32768.0, 16000)
        alen = amfcc.shape[0]
        if (alen < 240):
            amfcc = np.concatenate((amfcc, np.zeros(shape=((240 - alen), amfcc.shape[1]))), axis=0)
            
        elif (alen > 240):
            alen = 240
            amfcc = amfcc[:240]

    except Exception as e:
        print('axs')
        print('Error while processing audio: {} '.format(e))

    return amfcc, alen

def core(args):

    data_dir = args.data_dir
    workspace = args.workspace
    filename = args.filename
    model_type = args.model_type
    # model = args.model

    q = mp.Queue (maxsize=20)

    # Load data
    load_time = time.time()

    # tensorflow graph
    saver = tf.train.import_meta_graph(meta_path)   # import graph

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)           # graph = tf.get_default_graph()
        
        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        sequence_length_tensor = sess.graph.get_tensor_by_name(
            'vggish/input_sequence_length:0')
        output_tensor = sess.graph.get_tensor_by_name(
            'mix/prediction:0')

        mp.Process (target = audio_sample.core, args = (q,)).start()   # start audio sample
        combined = []
        _960ms = []
        while True:
            _960ms = q.get()
            print(q.qsize())
            while q.qsize() > 0:
                _960ms = q.get()

            combined += _960ms
            if (len(combined) / audio_sample.buf_size) >= _DURATION:
                # print('combined', len(combined))
                test_x_mfcc, test_x_len = raw2mfcc(combined)
                [[output], sl] = sess.run([output_tensor, sequence_length_tensor], feed_dict={features_tensor: [test_x_mfcc,], sequence_length_tensor: [test_x_len,]})       #output = model.predict(input)
                print_prediction(output, sl)
                combined[0:len(_960ms)] = []
