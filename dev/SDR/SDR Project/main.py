#public libraris and modules imports
import numpy as np
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sounddevice as sd
#import uhd

#local imports
import dsp
#import signalRecive as sr


#signal info
num_samps = int(1e6) 
freq = 99e6 
fs = 1e6 #sample rate
channels = [0] 
gain = 0 

#filter info
num_taps = 99 
cutoff = 75e3 #(Hz)

FFT_size = 1024

#inits
'''
usrp = sr.Usrp(freq,fs, gain)
usrp.initStreamer(channels)
usrp.startStream()
metadata = uhd.types.RXMetadata()


while(True):
    samples = np.zeros(int(fs), dtype = np.complex64) 
    sampels = usrp.recvSignal(samples, metadata)

    samples = dsp.LPfilter(samples,fs, num_taps, cutoff) 
    demod_samples = dsp.fm_demod(samples)

    demod_samples = dsp.decimation(demod_samples,25) #down sampling by 25 (40 Khz)
    sd.play(demod_samples, int(44.1e3)) 
'''
samples = np.fromfile(r"./samples.dat", dtype = np.complex64)
samples = dsp.LPfilter(samples,fs, num_taps, cutoff) 
demod_samples = dsp.fm_demod(samples)

demod_samples = dsp.decimation(demod_samples,25) #down sampling by 25 (40 Khz)
#sd.play(demod_samples, int(44.1e3))
write("test.wav", int(44.1e3), demod_samples.astype(np.float32))
