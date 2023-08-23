#imports
import numpy as np
from scipy import signal

def next_power_of_two(x): 
    return 1 if x==0 else 2**(x-1).bit_length()

def add_zeros(x): 
    N = len(x)
    powr = next_power_of_two(N)
    if (N != powr):
        zeros = np.zeros(powr,dtype = np.complex)
        zeros[:N] = x
        x = zeros
    return x

def fft(x): #mathmatical fft

    N = len(x)
    if N==1:
        return x
 

    X_even = fft(x[::2]) 
    X_odd = fft(x[1::2]) 
        
    factor = np.exp(-2j * np.pi * np.arange(N) / N) 
    return np.concatenate([X_even+factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd]) 

def spectrum(x, size): #create a spectrum using fft
    i = 0
    x = np.array_split(x,len(x)//size) 
    
    x = [a for a in x if a.size == size] 
    
    for a in x: 
        np.fft.fft(a) 
        a = np.power(np.abs(a),2) 
        x[i] = a
    
    x = np.average(x, axis = 0) 
    return x
    

    
def fftshift(x):
    N = len(x)
    return np.concatenate((x[N//2: ], x[:N//2])) 

def ifft(x):
    N = len(x)

    return 1/N*np.conj(fft(np.conj(x))) 

def fm_demod(x):# fm demodulazation 0.5*arctan(x * conj(xn-1))

    y = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))
    return y

def decimation(x, m):# downsampling x by m
    return x[::m]

def LPfilter(s, fs, num_taps, cutoff): #low pass filtering
    h = signal.firwin(num_taps, cutoff, nyq=fs/2) #creating filter
    s_fil= np.convolve(s,h) #applying filter
    return s_fil