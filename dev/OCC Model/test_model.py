from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
import feature_extraction as fe
import numpy as np 
import os
import librosa
import pickle
import math

dirname = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\Data Sets\Test"

tags = ["other","music"]

path = r".\model.model"

model = pickle.load(open(path, "rb"))

index = {tags[0]: -1, tags[1]: 1}
rates = {tags[0]:0,tags[1]:0}
for tag in tags:
    #inits
    guesses = dict.fromkeys(tags,0)
    tagdir = fr"{dirname}\{tag}"
    total = len(os.listdir(tagdir))
    print(f"{tag}:")
    for filename in os.listdir(tagdir):
        print(f"\t{filename}", end="")
        x, sr = librosa.load(fr"{tagdir}\{filename}", sr=44100)
        '''
        RMS=math.sqrt(np.mean(x**2))
        noise=np.random.normal(0, RMS, x.shape[0])
        x = x+noise
        '''
        audio_features = fe.extractFeatures(x[:10*sr],sr)
        guess = model.predict(np.array(audio_features).reshape(1, -1))
        if guess == -1:
            key_list = list(index.keys())
            val_list = list(index.values())
            position = val_list.index(guess)
            guesses[key_list[position]] += 1
            print(" speech")
        else:
            key_list = list(index.keys())
            val_list = list(index.values())
            position = val_list.index(guess)
            guesses[key_list[position]] += 1
            print(" music")
    rates[tag] = guesses[tag]/total
