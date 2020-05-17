import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import numpy as np
from numpy import argmax
import pandas as pd
import random
import os

import warnings
warnings.filterwarnings('ignore')

from classification_ffnn import classify_nn
from keras import optimizers
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

def resize_ps(ps):
    resize_ps = np.zeros((128, 600))
    if ps.shape[1] >= 600:
        resize_ps = ps[:,:600]
    else:
        resize_ps[:,:ps.shape[1]] = ps
    return resize_ps

def classify_model1(m1,y,sr):
    mel_spec = mel_spectogram(y,sr)
    mel_spec = resize_ps(mel_spec)
    mel_spec = mel_spec.reshape(1,128, 600, 1)
    classification= m1.predict(mel_spec);
    idx = np.argmax(classification)
    return idx

def mel_spectogram(y,sr):
    # mel-spectrogram parameters
    n_fft=2048
    hop_length=1024
    n_mels=128

    melgram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    ret = librosa.power_to_db(melgram, ref=np.max)
    return ret

def classify_song(filename_in):
    fileName, fileExtension = os.path.splitext(filename_in)
    filename_out = fileName + '-trimmed.wav'
    y, sr = librosa.load(filename_in)
    duree = librosa.get_duration(y=y, sr=sr)
    if duree>=60:
        y, sr = librosa.load(filename_in, offset=10, duration=duree-20)
        librosa.output.write_wav(filename_out, y, sr)
        num_tracks = int((duree-30)//30)
    elif duree >=30:
        librosa.output.write_wav(filename_out, y, sr)
        num_tracks = 1
    else:
        num_tracks = 0
    list_tracks = []
    m1 = load_model('../models/cnn.h5')
        # load json and create model
    json_file = open('../models/ffnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../models/ffnn.h5")
    print("Loaded model from disk")

    scaler = joblib.load('../models/scaler.save')

    opt = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
    loaded_model.compile(optimizer=opt,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    for i in range(num_tracks):
        offset = 30*i
        y_i, sr_i = librosa.load(filename_out, offset=offset, duration=30)
        list_tracks.append((y_i,sr_i))
        list_result = []
    for i in range(num_tracks):
        idx = classify_model1(m1,list_tracks[i][0],list_tracks[i][1])
        if idx == 2:
            idx = classify_nn(scaler,loaded_model,list_tracks[i][0],list_tracks[i][1])
        list_result.append(idx)
        list_result_count = [list_result.count(0),list_result.count(1),list_result.count(2),list_result.count(3),list_result.count(4)]
        percent_result = 100*np.divide(list_result_count,num_tracks)

    if num_tracks == 0:
        final = 'Impossible to classify, too short'
    else:
        final = 'Classical: {:.2f}%, Jazz: {:.2f}%, Hiphop: {:.2f}%, Reggae: {:.2f}%, Rock: {:.2f}%'.format(percent_result[0],percent_result[1],percent_result[2],percent_result[3],percent_result[4])

    os.remove(filename_out)
    return final
    #     class_pred_idx = np.argmax(list_result_count)
    # if class_pred_idx == 0 :
    #     class_pred = 'Classical'
    # elif class_pred_idx == 1 :
    #     class_pred = 'Jazz'
    # elif class_pred_idx == 2 :
    #     class_pred = 'Hiphop'
    # elif class_pred_idx == 3 :
    #     class_pred = 'Reggae'
    # elif class_pred_idx == 4 :
    #     class_pred = 'Rock'
    # return class_pred
