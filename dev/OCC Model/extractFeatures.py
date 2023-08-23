import numpy as np
import pandas as pd
import librosa
import os 
import warnings
import feature_extraction as fe
import sounddevice as sd
import math

secs = 1
tags = ["music"]
#tags = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
#columns = ["file_name","G#", "G", "F#", "F", "E", "D#", "D", "C#", "C", "B", "A#", "A", "zero_crossing","stft_var", "spectral_centroid_var","spectral_centroid_mean","spectral_rolloff_var","spectral_rolloff_mean","spectral_bandwidth_var","spectral_bandwidth_mean","mfcc1_var","mfcc1_mean","mfcc2_var","mfcc2_mean","mfcc3_var","mfcc3_mean","mfcc4_var","mfcc4_mean","mfcc5_var","mfcc5_mean","mfcc6_var","mfcc6_mean","mfcc7_var","mfcc7_mean","mfcc8_var","mfcc8_mean","mfcc9_var","mfcc9_mean","mfcc10_var","mfcc10_mean","mfcc11_var","mfcc11_mean","mfcc12_var","mfcc12_mean","mfcc13_var","mfcc13_mean","mfcc14_var","mfcc14_mean","mfcc15_var","mfcc14_mean","mfcc16_var","mfcc16_mean", "mfcc17_var","mfcc17_mean", "mfcc18_var","mfcc18_mean","mfcc19_var","mfcc19_mean","mfcc20_var","mfcc20_mean", "tag"]
dirname = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\Data Sets\Data"
columns = ["file_name","zcrr","ste","sf","tag"]

df = pd.DataFrame(columns = columns)
#df = pd.read_csv(r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\Data Sets\Data\audio_features.csv")

for tag in tags:
    tagdir = fr"{dirname}\to-add"
    print(f"{tag}:")
    for filename in os.listdir(tagdir):
        try:
            x, sr = librosa.load(fr"{tagdir}\{filename}", sr=44100)
            '''
            RMS=math.sqrt(np.mean(x**2))
            noise=np.random.normal(0, RMS, x.shape[0])
            x = x+noise
            '''
            subs = [x[i:i+secs*sr] for i in range(0, len(x), secs*sr)]
            if len(subs[-1]) != secs*sr:
                subs = subs[:-1]
            i = 0
            for sub in subs:
                features = [f"{filename} part:{i+1}"]

                audio_features = fe.extractFeatures(sub,sr)
                features = features + audio_features
                features.append(tag)
                s = pd.Series(features, index = columns)
                df = df.append(s, ignore_index = True)
                print(f"\t{filename} part:{i+1}")
                i+=1
            #print(f"\t{i}")
        except Exception as err:
            print(err)
df.to_csv("./audio_features1.csv")

