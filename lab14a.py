import pickle
import numpy
import math
from wav_utils import *
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

with open('signal.pkl', 'rb') as f:
    signal = pickle.load(f)

dt = 1.0e-7
rate = int(1/dt)

n = len(signal)
seconds = int(n / rate)
sample_rate = 44100
interleave = int(rate / sample_rate)
max_frequency = 5000    # We are told the signal is in the the range 0-5Khz
cutoff = int(seconds * max_frequency)

# Helper parameters for plotting:
show_plots = False      
pn = 1000

def frequency_from_index(idx):
    assert(idx >= 0 and idx <= 23)
    return 560e+3 + idx * 10e+3

def extract_signal(carrier_frequency, signal):

    omega = carrier_frequency * 2 * numpy.pi

    def calc_factor(x):
        return np.cos(omega * x * dt)

    index_array = numpy.arange(n)
    cosine_factors = calc_factor(index_array)
    multiplied_signal = cosine_factors * signal

    if show_plots == True:
        plt.plot(index_array[:pn], signal[:pn], index_array[:pn], cosine_factors[:pn], index_array[:pn], multiplied_signal[:pn])
        plt.show()

    signal_fft = fft(multiplied_signal)

    if show_plots == True:
        plt.plot(numpy.real(signal_fft))
        plt.show()

    print("Zeroing entries after the first {0}.".format(cutoff))
    signal_fft[cutoff:] = 0
    signal_fft[0] = 0    # remove bias as well.

    modified_signal = ifft(signal_fft)
    modified_signal = numpy.real(modified_signal)

    return modified_signal

def wave_from_signal(source):

    subsampled = source[::interleave]

    if show_plots == True:
        plt.plot(subsampled[0:1000])
        plt.show()


    max_value = numpy.amax(subsampled)
    min_value = numpy.amin(subsampled)

    def renormalize(x):
        return 2 * (x - min_value) / (max_value - min_value) - 1

    subsampled = renormalize(subsampled)

    return subsampled


for i in range(0, 24):
    print ("Processing song...", i)
    carrier_frequency = frequency_from_index(i)
    result = extract_signal(carrier_frequency, signal)
    wav = wave_from_signal(result)
    out_file = "song{0}.wav".format(i)
    wav_write(wav, sample_rate, out_file)


