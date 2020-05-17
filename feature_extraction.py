#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:09:46 2019
@author: Pedro, Rodrigo, Alix
"""

# feature extractoring and preprocessing data
import numpy as np
#import essentia.standard as es
# matplotlib inline
import os
import csv
import librosa

import warnings
warnings.filterwarnings('ignore')

#header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate tempo'
header  = 'filename spectral_centroid rmse rolloff zcr tempo'
n_mfcc = 10
#n_gfcc = 1
for i in range(1, n_mfcc+1):
    header += f' mfcc{i}'
for i in range(1, 13):
    header += f' chroma_stft{i}'
header += ' label'
header = header.split()

file = open('features/data_selected.csv', 'w', newline='')

#file = open('../features_csv/data_reggae_hiphop.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

#genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

genres = 'classical hiphop jazz reggae rock'.split()

sr = 22050

#Create the objects to extract features using Essentia
#spectrum_extractor = es.Spectrum()
#danceability_extractor = es.Danceability(sampleRate=sr)
#pitch_salience_extractor = es.PitchSalience(sampleRate=sr)
#zcr_extractor = es.ZeroCrossingRate()
#mfcc_extractor = es.MFCC(sampleRate=sr,numberCoefficients=n_mfcc, highFrequencyBound = sr/2)
#gfcc_extractor = es.GFCC(numberCoefficients=n_gfcc, sampleRate=sr, highFrequencyBound = sr/2)
#BPM_Hist_extractor = es.BpmHistogramDescriptors()

for g in genres:
    for filename in os.listdir(f'dataset/{g}'):
        songname = f'dataset/{g}/{filename}'
        audio, _ = librosa.load(songname, mono=True, duration=30)

        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rmse = librosa.feature.rms(y=audio)
#        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

#       Add TEMPO features
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = tempo.item()
#        prior = scipy.stats.uniform(40, 200)
#        utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
#        utempo = utempo.item()


######################################################################################################
################################## ESSENTIA ##########################################################
######################################################################################################
        # Add Features extracted from Essentia
#        To calculate the spectrum we need a pair number of samples, so we
#        delete the last sample if it is impair
#        audio = es.MonoLoader(filename=songname,sampleRate=sr)() # Load song to use ESSENTIA library
#        if(np.size(audio,0)%2!=0):
#            audio = audio[:-1]

#        spectrum = spectrum_extractor(audio)

#        We use BPM to extract rythm features, the main beat of the song
#        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
#        bpm, _, _, _, beats_intervals = rhythm_extractor(audio)

#        Detrended Fluctuation Analysis of Music Signals - Different transforms
#        danceability,_ = danceability_extractor(audio)

#        We use pitch salience to check the correlation of the song, i.e if the song is similar to
#        itself in the extract we have. Repeating songs will have higher values of pitch salience
#        pitch_salience = pitch_salience_extractor(spectrum)

#        We use the MFCC
#        _, mfcc = mfcc_extractor(spectrum)

#        We use the zero crossing rate to count the number of times the sample passed the 0
#        zcr = zcr_extractor(audio)


#        firstPeakBPM, firstPeakWeight, firstPeakSpread, _, _, _, histogram = BPM_Hist_extractor(beats_intervals)
#        _,gfcc = gfcc_extractor(spectrum)

#        hfc_extract = es.HFC(sampleRate=sr)
#        hfc = hfc_extract(spectrum)

#        peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = es.BpmHistogramDescriptors()(beats_intervals)


#        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)} {np.mean(utempo)}'

        to_append = f'{filename} {np.mean(spec_cent)} {np.mean(rmse)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(tempo)}'

        for e in mfcc:
            to_append += f' {np.mean(e)}'

        for e in chroma_stft:
            to_append += f' {np.mean(e)}'

        to_append += f' {g}'

        file = open('features/data_selected.csv', 'a', newline='')
#        file = open('../features_csv/data_reggae_hiphop.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
