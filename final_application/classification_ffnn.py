# -*- coding: utf-8 -*-
import numpy as np
import librosa

import warnings
warnings.filterwarnings('ignore')

def classify_nn(scaler,loaded_model,audio,sr):

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

    param = scaler.transform(param)

    result = loaded_model.predict(param)

    result = np.argmax(result,axis=1)

    if result == 0:
        label = 'Hiphop'
        idx = 2
    elif result == 1:
        label = 'Reggae'
        idx = 3
    else:
        label = 'Rock'
        idx = 4

    return idx
