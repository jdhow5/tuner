# ## Polyphonic Tuner 

# This project will implement a polyphonic tuner using audio recorded from a smartphone.

import numpy as np
import scipy.io.wavfile as wave
from scipy.signal import butter, filtfilt, lfilter, get_window
from gammatone import filters as gt
import matplotlib.pyplot as plt

def clip_audio(data, time) :
    #analyze a 93ms window after 100 ms to avoid noise and clip audio for each frame
    clip_samp = fs*time/1000
    analysis_samp = clip_samp + round(fs*93/1000)
    trimmed_audio = data[int(clip_samp):int(analysis_samp)]
    return trimmed_audio

def standard_deviation(transform) :
    K = len(transform)
    V = 0
    for i in range(1, K):
        V += (transform[i]*transform[K-i])**2
    V += transform[0]**2 
    sd = np.sqrt(V / (K - 1))
    return sd

def compression(data) :
    transform = np.fft.fft(data, 2*len(data))
    sigma = standard_deviation(transform)
    transform = transform * (sigma**(0.33-1))
    compressed = np.fft.ifft(transform)
    return compressed

def half_wave_rectification(data) :
    rectify = np.clip(data, 0, None)
    return rectify

def butter_lfilter(data, fc, fs, order=5, fb = False):
    nyq = 0.5 * fs
    low = fc / nyq
    b, a = butter(order, low, btype='low')
    if fb:
        y = filtfilt(b, a, data)
    else: 
        y = lfilter(b, a, data)
    
    return y

#########SOURCED FROM: https://github.com/gburlet/multi-f0-estimation#########

def search_smax(data, fs, max_f, min_f, tau_prec=1.0) :
    Q = 0           # index of the new block
    q_best = 0      # index of the best block

    tau_low = [round(fs/2100)] # in samples/cycle
    tau_up = [round(fs/60)]  # in samples/cycle
    smax = [0]

    while tau_up[q_best] - tau_low[q_best] > tau_prec:
        # split the best block and compute new limits
        Q += 1
        tau_low.append((tau_low[q_best] + tau_up[q_best])/2)
        tau_up.append(tau_up[q_best])
        tau_up[q_best] = tau_low[Q]

        # compute new saliences for the two block-halves
        for q in [q_best, Q]:
            salience, _ = calc_salience(data, fs, tau_low[q], tau_up[q])
            if q == q_best:
                smax[q_best] = salience
            else:
                smax.append(salience)

        q_best = np.argmax(smax)

    # estimated fundamental period of the frame
    tau_hat = (tau_low[q_best] + tau_up[q_best])/2

    # calculate the spectrum of the detected fundamental period and harmonics
    salience_hat, harmonics = calc_salience(data, fs, tau_low[q_best], tau_up[q_best])
    K = len(data)<<1
    detected = calc_harmonic_spec(fs, K, harmonics)

    return tau_hat, salience_hat, detected
    
def calc_salience(data, fs, tau_low, tau_up) :
    salience = 0 

    tau = (tau_low + tau_up)/2
    delta_tau = tau_up - tau_low

    # calculate the number of harmonics under the nyquist frequency
    # the statement below is equivalent to floor((fs/2)/fo)
    num_harmonics = int(np.floor(tau/2))

    # calculate all harmonic weights
    harmonics = np.arange(num_harmonics)+1
    g = (fs/tau_low + 52) / (harmonics*fs/tau_up + 320)

    # calculate lower and upper bounds of partial vicinity
    nyquist_bin = len(data)
    K = nyquist_bin<<1
    lb_vicinity = K/(tau + delta_tau/2)
    ub_vicinity = K/(tau - delta_tau/2)

    # for each harmonic
    harmonics = []
    for m in range(1,num_harmonics+1):
        harmonic_lb = round(m*lb_vicinity)
        harmonic_ub = min(round(m*ub_vicinity), nyquist_bin)
        harmonic_bin = np.argmax(data[harmonic_lb-1:harmonic_ub]) + harmonic_lb-1
        harmonic_amp = data[harmonic_bin]
        w_harmonic_amp = g[m-1] * harmonic_amp

        # save the properties of this fundamental period and harmonics
        harmonics.append({'bin': harmonic_bin, 'amp': w_harmonic_amp})

        salience += w_harmonic_amp

    return salience, harmonics

def calc_harmonic_spec(fs, K, harmonics) :
    nyquist_bin = K>>1
    # initialize spectrum of detected harmonics
    detected = np.zeros(nyquist_bin)

    # calculate the partial spectrum for each harmonic
    # Klapuri PhD Thesis, page 62 and (Klapuri, 2006) Section 2.5
    # Even with these sources, the algorithm for estimating the 
    # spectrum of the fundamental and partials is rather unclear.
    frame_len_samps = int(fs * 0.093)
    win = get_window('hanning', frame_len_samps) 
    window_spec = np.abs(np.fft.fft(win, K))
    partial_spectrum = np.hstack((window_spec[10::-1],
                                window_spec[1:10+1]))
    # normalize the spectrum
    partial_spectrum /= np.max(partial_spectrum)

    for h in harmonics:
        h_lb = max(0, h['bin']-10)
        h_ub = min(nyquist_bin-1, h['bin']+10)

        # translate the spectrum of the window function to the position of the harmonic
        detected[h_lb:h_ub+1] = h['amp']*partial_spectrum[h_lb-h['bin']+10:h_ub-h['bin']+10+1]

    return detected

############################################################################

def choose_best(notes, freqs) :
    for i in range(len(freqs)):
        if freqs[i] > 72 and freqs[i] < 92:
            if np.absolute(82.41-freqs[i]) < np.absolute(82.41-notes[0]):
                notes[0] = freqs[i]
        elif freqs[i] > 100 and freqs[i] < 120:
            if np.absolute(110.0-freqs[i]) < np.absolute(110.0-notes[1]):
                notes[1] = freqs[i]
        elif freqs[i] > 136 and freqs[i] < 156:
            if np.absolute(146.83-freqs[i]) < np.absolute(146.83-notes[2]):
                notes[2] = freqs[i]
        elif freqs[i] > 186 and freqs[i] < 206:
            if np.absolute(196.0-freqs[i]) < np.absolute(196.0-notes[3]):
                notes[3] = freqs[i]
        elif freqs[i] > 236 and freqs[i] < 256:
            if np.absolute(246.94-freqs[i]) < np.absolute(246.94-notes[4]):
                notes[4] = freqs[i]
        else:
            if np.absolute(329.63-freqs[i]) < np.absolute(329.63-notes[5]):
                notes[5] = freqs[i]
    return notes


num_freqs = 72
lowf = 60 #Hz
highf = 2100 #Hz
d = 0.89 #amount to remove detected signal from residual
time = 100 #time to start analyzing audio 
delta_time = 100 # time between analysis frames
notes = np.zeros(6, dtype=float)

input_signal = wave.read('16kHz_acTuned.wav')
fs = input_signal[0]
T = 1/fs
audio = np.asarray(input_signal[1])

while time < (len(audio)-fs*time/1000):
    trim_audio = clip_audio(audio, time)

    center_freqs = gt.erb_space(lowf, highf, num_freqs)
    filt_coefs = gt.make_erb_filters(fs, center_freqs)
    channels = gt.erb_filterbank(trim_audio, filt_coefs)

    process_chan = np.empty([len(channels), len(channels[0])*2], dtype=complex)
    mag_chan = np.empty_like(process_chan, dtype=float)
    
    #Cochlea simulation
    for idx in range(len(channels)):
        process_chan[idx,:] = compression(channels[idx,:])
        process_chan[idx,:] = half_wave_rectification(process_chan[idx,:])
        low_cutoff = center_freqs[idx]*1.5
        process_chan[idx,:] = butter_lfilter(process_chan[idx,:], low_cutoff, fs)
        mag_chan[idx,:] = np.absolute(np.fft.fft(process_chan[idx,:]))

    #summation of DFT magnitudes across each channel  
    U = np.sum(mag_chan, axis=0)
    U_R = np.array(U[1:int(len(U)/2)]) #residual with removed symmetry (negative terms)

    f = np.fft.fftfreq(len(U), d=T)
    freqs = np.array(f[1:int(len(f)/2)])
    index_max = np.argmax(freqs>700)

    sal_hats = []
    fund_freqs = []

    #calculate saliences/fundamental frequencies and remove harmonic spectrum from signal
    while(len(sal_hats) < 6):
        tau_hat, sal_hat, U_D = search_smax(U_R, fs, highf, lowf, tau_prec=0.5)
        sal_hats.append(sal_hat)
        fund_freqs.append(fs/tau_hat)
        U_R -= d*U_D
    
    #take best estimates from each frame and update result
    notes = choose_best(notes, fund_freqs)
    time+=delta_time

print(notes)

