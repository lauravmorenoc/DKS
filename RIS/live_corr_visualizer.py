import adi            # Gives access to pluto commands
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from graphics import *


''' Functions '''

def conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,buffer_size):
    '''Configure properties for the Radio'''
    sdr.sample_rate = int(samp_rate)
    sdr.rx_rf_bandwidth = int(fc0 * 3)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain)
    sdr.rx_buffer_size = int(NumSamples)
    sdr._rxadc.set_kernel_buffers_count(1)  # Set buffers to 1 to avoid stale data on Pluto
    fs=int(sdr.sample_rate)
    ts=1/fs
    return [fs, ts]

def get_fft(data):
    Rx_0=data
    NumSamples = len(Rx_0)
    win = np.hamming(NumSamples)
    y = Rx_0 * win
    sp = np.absolute(np.fft.fft(y))
    sp = sp[1:-1]
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / (np.sum(win)/2)    # Scale FFT by window and /2 since we are using half the FFT spectrum
    s_dbfs = 20*np.log10(s_mag/(2**12))     # Pluto is a 12 bit ADC, so use that to convert to dBFS
    xf = np.fft.fftfreq(NumSamples, ts)
    xf = np.fft.fftshift(xf[1:-1])/1e6
    return [xf, s_dbfs]

def show_fft(xf, s_dbfs):
    plt.clf()  # Clear the previous plot
    plt.plot(xf, s_dbfs, label=f"Scan {i+1}")  # Update plot
    plt.ylim([-100, 0])
    #plt.set_
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("dBfs")
    plt.legend()
    plt.xlim([-0.25, 0.25])
    plt.draw()
    plt.grid()
    plt.pause(0.01)  # Pause to allow real-time update


''' Variables '''

samp_rate = 5.3e5    # must be <=30.72 MHz if both channels are enabled (530000)
NumSamples = 300000 # buffer size (4096)
rx_lo = 5.3e9
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain = 0 # 0 to 50 dB
tx_lo = rx_lo
tx_gain = -50 # -90 to 0 dB
fc0 = int(200e3)
RIS_seq_period=0.001 
err1=[1]
err2=[1]

''' Control Variables '''
threshold_factor=3
num_av_corr=3
downsample_factor=30
th_cycles=10

'''Create Radios'''

'''sdr=adi.ad9361(uri='ip:192.168.2.1')'''
sdr=adi.ad9361(uri='usb:1.8.5')
[fs, ts]=conf_sdr(sdr, samp_rate, fc0, rx_lo, rx_mode, rx_gain,NumSamples)

''' Pre-designed sequence '''

mseq1=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
mseq2=np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
M=len(mseq1)
amp=1 # signal amplitude
mseq1= np.where(mseq1 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
mseq2= np.where(mseq2 == 0, amp, -amp) # rearange sequence so 0=amp, 1=-amp
sps=17000/downsample_factor
mseq_upsampled1=np.repeat(mseq1, sps) # upsample sequence
mseq_upsampled2=np.repeat(mseq2, sps) # upsample sequence
M_up = M*sps

'''Collect data'''

for r in range(20):    # grab several buffers to give the AGC time to react (if AGC is set to "slow_attack" instead of "manual")
    data = sdr.rx()

scanning_time=30 # seconds
pause=0.00001
scanning_ts=pause+NumSamples*ts
num_scans=int(scanning_time/scanning_ts)
peaks=[-60]

fig=plt.figure()
bx=fig.add_subplot(111)
line1, = bx.plot([],[],label="RIS #1",marker='*')
line2, = bx.plot([],[],label="RIS #2")
line3,=bx.plot([],[],label="TH1")
line4,=bx.plot([],[],label="TH2")
bx.legend()
fig2=plt.figure()
cx=fig2.add_subplot(111)
line5, = cx.plot([],[],label="error 1")
line6, = cx.plot([],[],label="error 2")
cx.legend()
s=0

''' Finding threshold '''

corr_peaks_first_seq=[0]
corr_peaks_second_seq=[0]
corr_final_first_seq=[0]
corr_final_second_seq=[0]

for j in range(th_cycles):
    data = sdr.rx()
    Rx = data[0]
    Rx=Rx[::downsample_factor]
    envelope=np.abs(Rx)/2**12
    env_mean=np.mean(envelope)
    envelope-=env_mean
    envelope=envelope/np.max(envelope)
    corr_peaks_first_seq=np.abs(correlate(mseq_upsampled1, envelope, mode='full'))/M_up # normalized
    corr_peaks_second_seq=np.abs(correlate(mseq_upsampled2, envelope, mode='full'))/M_up # normalized
    corr_final_first_seq=np.append(corr_final_first_seq, np.max(corr_peaks_first_seq))
    corr_final_second_seq=np.append(corr_final_second_seq, np.max(corr_peaks_second_seq))

th_1=threshold_factor*np.mean(corr_final_first_seq[1:])
th_2=threshold_factor*np.mean(corr_final_second_seq[1:])
corr_av_1=0
corr_av_2=0
for i in range(1000):
    corr_final_first_seq=[0]
    corr_final_second_seq=[0]
    del data
    for j in range(num_av_corr):
        corr_peaks_first_seq=[0]
        corr_peaks_second_seq=[0]
        data = sdr.rx()
        Rx = data[0]
        Rx=Rx[::downsample_factor]
        envelope=np.abs(Rx)/2**12
        env_mean=np.mean(envelope)
        envelope-=env_mean
        envelope=envelope/np.max(envelope)
        corr_peaks_first_seq=np.abs(correlate(mseq_upsampled1, envelope, mode='full'))/M_up # normalized
        corr_peaks_second_seq=np.abs(correlate(mseq_upsampled2, envelope, mode='full'))/M_up # normalized
        corr_final_first_seq=np.append(corr_final_first_seq, np.max(corr_peaks_first_seq))
        corr_final_second_seq=np.append(corr_final_second_seq, np.max(corr_peaks_second_seq))

    corr_av_1=np.append(corr_av_1,np.mean(corr_final_first_seq[1:]))
    corr_av_2=np.append(corr_av_2,np.mean(corr_final_second_seq[1:]))
    
    RIS_1 =0
    RIS_2=0

    RIS_1_state= np.abs((np.mean(corr_final_first_seq[1:])>th_1)*1)
    RIS_2_state= np.abs((np.mean(corr_final_second_seq[1:])>th_2)*1)
    err1=np.append(err1,np.abs(RIS_1-(np.mean(corr_final_first_seq[1:])>th_1)*1))
    err2=np.append(err2,np.abs(RIS_2-(np.mean(corr_final_second_seq[1:])>th_2)*1))

    '''AK'''
    if(len(time)>fs*5):
     time =time[len(time)-fs*5:]
    
    line1.set_xdata(time)
    line1.set_ydata(corr_av_1)
    line2.set_xdata(time)
    line2.set_ydata(corr_av_2)
    line3.set_xdata(time)
    line3.set_ydata([th_1]*len(corr_av_1))
    line4.set_xdata(time)
    line4.set_ydata([th_2]*len(corr_av_2))
    line5.set_xdata(time)
    line5.set_ydata(err1)
    line6.set_xdata(time)
    line6.set_ydata(err2)
    bx.set_ylim([0,1.2*np.max([np.max(corr_av_1), np.max(corr_av_2),th_1,th_2])])
    bx.set_xlim([0,np.max(time)])
    cx.set_ylim([-1,2])
    cx.set_xlim([0,np.max(time)])
    plt.pause(pause)  # Pause to allow real-time update

plt.show()  # Show final figure (if needed)'''

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done