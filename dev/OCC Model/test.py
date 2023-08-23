r'''
import librosa
import pandas as pd
import numpy as np
import os
import warnings
import math 
warnings.filterwarnings("ignore")
path = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\Data Sets\Test"
tags = ["other","music"]
lists = []
sr = 44100

for tag in tags:
    tagdir = fr"{path}\{tag}"
    print(f"{tag}:")
    l = []
    for file in os.listdir(tagdir)[:20]:
        try:
            x, sr = librosa.load(fr"{tagdir}\{file}", sr=sr)
            SCs = np.mean(librosa.feature.spectral_centroid(x,sr=sr))
            print(f"{file}:{SC}")
            l.append(SC)
        except Exception as e: print(e)
    lists.append(l)
print(f"flux mean for non-music:{np.mean(lists[0])}")
print(f"flux mean for music:{np.mean(lists[1])}")
'''

import feature_extraction as fe
import numpy as np 
import os
import librosa
import pickle

secs = 10
data_path = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\LOF Model\test.wav"
path = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\LOF Model\model.model"
model = pickle.load(open(path, "rb"))
x,sr = librosa.load(data_path,sr = 44100)
subs = [x[i:i+secs*sr] for i in range(0, len(x), secs*sr)]
if len(subs[-1]) != secs*sr:
    subs = subs[:-1]
for sub in subs:
    audio_features = fe.extractFeatures(sub,sr)
    guess = model.predict(np.array(audio_features).reshape(1, -1))
    print(guess)

