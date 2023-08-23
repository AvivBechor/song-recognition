import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")


def zero_crossing_rate_raio(x,sr,wlen = 512): #zero crossing rate ratio (standard diviation)
    zcrrs = librosa.feature.zero_crossing_rate(x + 0.0001,hop_length= wlen)
    std = np.std(zcrrs[0])
    return std
def short_time_energy(x,sr,wlen = 512):#short time energy (coefficient of variation)
    frames = np.array_split(x,wlen)
    ste = [np.sum(np.power(np.abs(frame),2))/(wlen*(frame.size/wlen)) for frame in frames ]
    cv = np.std(ste)/np.mean(ste)
    return cv

def spectral_flux(x,sr):
    onset = librosa.onset.onset_strength(y = np.abs(np.fft.fft(x)),sr=sr)
    sf = np.linalg.norm(np.diff(onset))
    return sf

def sc(x,sr,wlen = 512): #spectral centroid
    sc = np.mean(librosa.feature.spectral_centroid(x,sr,wlen))
    return sc
def extractFeatures(x,sr,wlen = 512):
    audio_features = []
    #extracting features
    std = zero_crossing_rate_raio(x,sr,wlen)
    cv = short_time_energy(x,sr,wlen)
    sf = spectral_flux(x,sr)
    #sc = sc(x,sr,wlen)
    #appending features
    audio_features.append(std)
    audio_features.append(cv)
    audio_features.append(sf)
    #audio_features.append(sc)
    '''
    #chromagram
    chromagram = librosa.feature.chroma_stft(x, sr=sr)
    numberOfWindows = chromagram.shape[1]
    freqVal = chromagram.argmax( axis = 0 )
    histogram, _bin = np.histogram( freqVal, bins = 12 )
    normalized_hist = histogram.reshape( 1, 12 ).astype( float ) / numberOfWindows
    audio_features = normalized_hist.tolist()[0]

    #spectogram
    stfs = librosa.stft(x)
    audio_features.append(stfs.var())

    #spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(x, sr=sr)[0]
    audio_features.append(spectral_centroid.var())
    audio_features.append(spectral_centroid.mean())

    #spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    audio_features.append(spectral_rolloff.var())
    audio_features.append(spectral_rolloff.mean())

    #spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    audio_features.append(spectral_bandwidth.var())
    audio_features.append(spectral_bandwidth.mean())

    #MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    audio_features.extend(list(mfccs.var(axis = 1)))
    audio_features.extend(list(mfccs.mean(axis = 1)))
    '''
    return audio_features
