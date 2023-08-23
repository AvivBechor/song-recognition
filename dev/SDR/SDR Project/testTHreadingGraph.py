lock = threading.Lock() 


def plotGraph(): 

    plt.ion() 
    fig = plt.figure()  
    ax1 = plt.subplot(111) 
    line1, = ax1.plot(np.abs(FFT.fftshift(FFT.spectrum(samples,FFT_size)))) 

    while(True): 
        if(not lock.locked()): 
            line1.set_ydata(np.abs(FFT.fftshift(FFT.spectrum(samples,FFT_size)))) 
            fig.canvas.draw() 
            fig.canvas.flush_events()
