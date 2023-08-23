#public libraris and modules imports
import numpy as np
import uhd
import math
import os
import pickle
#local imports
import dsp
import signalRecive as sr
import feature_extraction as fe

answers = {-1:"music",1:"speech"}
np.errstate(divide = "ignore")
with open('./parameters.txt') as f:
    lines = f.readlines()
    f.close()

for line in range(len(lines)):
    if(lines[line]!=""):
        lines[line] = lines[line].split(':')[1].split('\n')[0]

#signal info

fs = int(200e3) 
freqs = np.fromiter((range(int(88.1e6),int(108.1e6),fs)),dtype = float)
time = 1 #secs
num_samps = fs  * time
'''
num_samps = 1024
freq = float(lines[0]) 
fs = float(lines[1]) #sample rate
'''
channels = [0] 
gain = int(lines[2]) 

#filter info
num_taps = int(lines[3]) 
cutoff = float(lines[4]) #(Hz)
threshold = -82#dBm
lin_threshold = 10**((threshold - 30)/10)
FFT_size = 1024
ip = str(lines[5])
eps = 0.1

#inits
usrp = sr.Usrp(freqs[0], fs, gain, ip)
path = r"./noisy_model.model"
model = pickle.load(open(path, "rb"))
jumps = int(300e3)
power_norm = -32
#usrp.initStreamer(channels)
#usrp.startStream()
metadata = uhd.types.RXMetadata()
#dBm conversion : 10 ln(FFT(X)) + 30
freqs = np.fromiter((range(int(88e6),int(108e6),jumps)),dtype = float)
while(True):
    for freq in freqs:
        
        samples = usrp.usrp.recv_num_samps(num_samps, freq, fs,channels,gain)[0]
        
        
        power = dsp.power_spectrum_avg(samples)
        avg = 10*np.log10(power)+30  + power_norm
        print(avg)
        print(f"{freq/1000000}: ", end="")
        if (avg < threshold):
            continue
        audio_features = []
        samples = dsp.LPfilter(samples,fs, num_taps, cutoff) 
        demod_samples = dsp.fm_demod(samples)
        demod_samples = dsp.decimation(demod_samples,25) #down sampling by 25 (40 Khz)
        audio_features = fe.extractFeatures(demod_samples, 40e3)
        guess = model.predict(np.array(audio_features).reshape(1, -1))
        print(answers[guess[0]])
        
        
