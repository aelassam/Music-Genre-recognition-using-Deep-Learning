# -*- coding: utf-8 -*-
import numpy as np
import librosa

from keras import optimizers
from keras.models import model_from_json
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

def classify_nn(songname):

    audio, sr = librosa.load(songname, mono=True, duration=30)

    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rmse = librosa.feature.rms(y=audio)
#        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)

    n_mfcc = 10
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

#       Add TEMPO features
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo = tempo.item()

    param = [np.mean(spec_cent), np.mean(rmse), np.mean(rolloff), np.mean(zcr), np.mean(tempo)]

    for e in mfcc:
        param.append(np.mean(e))

    for e in chroma_stft:
        param.append(np.mean(e))

    param = np.transpose(param)
    param = param.reshape(1,-1)

    scaler = joblib.load('models/scaler.save')

    param = scaler.transform(param)

    # load json and create model
    json_file = open('models/ffnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/ffnn.h5")
    print("Loaded model from disk")

    opt = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
    loaded_model.compile(optimizer=opt,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    result = loaded_model.predict(param)

    result = np.argmax(result,axis=1)

    if result == 0:
        label = 'Hiphop'
    elif result == 1:
        label = 'Reggae'
    else:
        label = 'Rock'

    return result
