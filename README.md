# Polyphonic Guitar Tuner
A polyphonic guitar tuner written in Python.  The current tuner is based on Anssi Klapuri's 2005 paper "A Perceptually Motivated Multiple-F0 Estimatation Method" and his 2006 paper "Multiple Fundamental Frequency Estimation by Summing Harmonic Amplitudes".

The code contains salience calculation functions that are sourced from Gregory Burlet's Multiple-Fundamental Frequency Estimator: https://github.com/gburlet/multi-f0-estimation/

Jason Heeris' Gammatone Filterbank Toolkit is required to run the tuner: https://github.com/detly/gammatone 

A few audio files of a tuned acoustic guitar are provided to test the tuner.

# Future Work
I plan on implementing this tuner in Java to create a real-time polyphonic tuning app for Android.  The tuner currently provides accurate results when best estimates are chosen from each analysis frame, however the processing time needs to be improved.

# Author
Jacob Howard
2017
