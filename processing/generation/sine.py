import numpy as np

def generate_sine(amplitude, theta, f_analog, f_sampling, duration=1.0):

    if f_sampling < 2 * f_analog:
        raise ValueError("Sampling frequency must be at least 2× the analog frequency (Nyquist).")

    t = np.arange(0.0, duration, 1.0 / float(f_sampling))
    signal = amplitude * np.sin(2 * np.pi * f_analog * t + theta)

    indices = list(range(len(signal)))
    return indices, signal.tolist()
