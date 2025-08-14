import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
# Import the function from main.py
from old_main import SignalProcessorApp
def calculate_filter_order(transition_band, fs, window_type):
    delta_f = transition_band / fs
    if window_type == 'rectangular':
        N = np.ceil(0.9 / delta_f)
    elif window_type == 'hanning':
        N = np.ceil(3.1 / delta_f)
    elif window_type == 'hamming':
        N = np.ceil(3.3 / delta_f)
    elif window_type == 'blackman':
        N = np.ceil(5.5 / delta_f)
    
    if(N%2!=1):
        N = N + 1
    return int(N)

def design_filter(filter_type, fs, stopband_attenuation, fc, transition_band, window_type, F1, F2):
    # Calculate filter order N
    if stopband_attenuation <= 21:
        window_type = "rectangular"
    elif stopband_attenuation <= 44:
        window_type = "hanning"
    elif stopband_attenuation <= 53:
        window_type = "hamming"
    elif stopband_attenuation  <= 74:
        window_type = "blackman"
    N = calculate_filter_order(transition_band, fs, window_type)
    
    # Adjust N based on the window's constraints
    n = np.arange(-N//2 + 1, N//2 + 1)
    
    fc = (FC + (transition_band/2)) / FS
    f1 = (F1 - (transition_band/2)) / FS
    f2 = (F2 + (transition_band/2)) / FS
    
    filt = []
    index = []
    for i in n:
        if filter_type == "Low pass":
            if i != 0:
                h = 2 * fc * np.sin(i * 2 * np.pi * fc) / (i * 2 * np.pi * fc)
            else:
                h = 2 * fc
        elif filter_type == "High pass":
            if i != 0:
                h = -2 * fc * np.sin(i * 2 * np.pi * fc) / (i * 2 * np.pi * fc)
            else:
                h = 1 - 2 * fc
        elif filter_type == "Band pass":
            if i != 0:
                h1 = (2 * f2 * (np.sin(i * 2 * np.pi * f2) / (i * 2 * np.pi * f2)))
                h2 = (2 * f1 * (np.sin(i * 2 * np.pi * f1) / (i * 2 * np.pi * f1)))
                h = h1 - h2
            else:
                h = 2 * (f2 - f1)
        elif filter_type == "Band stop":
            if i != 0:
                h = 2 * f1 * np.sin(i * 2 * np.pi * f1) / (i * 2 * np.pi * f1) - 2 * f2 * np.sin(i * 2 * np.pi * f2) / (i * 2 * np.pi * f2)
            else:
                h = 1 - 2 * (f2 - f1)

        # Apply the window function
        if window_type == 'rectangular':
            w = 1
        elif window_type == 'hanning':
            w = 0.5 + 0.5 * np.cos((2 * np.pi * i) / N)
        elif window_type == 'hamming':
            w = 0.54 + 0.46 * np.cos((2 * np.pi * i) / N)
        elif window_type == 'blackman':
            w = 0.42 + 0.5 * np.cos((2 * np.pi * i) / (N - 1)) + 0.08 * np.cos((4 * np.pi * i) / (N - 1))


        filt.append(h * w)
        index.append(i)
    
    # Print filt and index values in the format: index value
    for i, val in enumerate(filt):
        print(f"{index[i]} {val}")
    #SignalProcessorApp.save_result("Filter Coefficient", index, filt)
    return filt, index

#SignalProcessorApp.convolve()
if __name__ == "__main__":
# Example usage:
    FilterType = "Band stop"
    FS = 1000
    FC = 500
    StopBandAttenuation = 60
    F1 = 150
    F2 = 250
    TransitionBand = 50
    WindowType = 'hamming'  # Can be 'rectangular', 'hanning', 'hamming', or 'blackman'

    filter_coefficients, index = design_filter(FilterType, FS, StopBandAttenuation, FC, TransitionBand, WindowType, F1, F2)


