#imports
import uhd

class Usrp:
    def __init__(self, freq, rate,gain):
            self.usrp = uhd.usrp.MultiUSRP("addr=192.168.10.5")
            self.usrp.set_rx_rate(rate) 
            self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq)) 
            self.usrp.set_rx_gain(gain) 
    
    def initStreamer(self, channels):
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = channels 
        self.streamer = self.usrp.get_rx_stream(st_args)

    def startStream(self):
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont) 
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd) 
        
    def recvSignal(self, buffer, metadata):
        return self.streamer.recv(buffer, metadata)