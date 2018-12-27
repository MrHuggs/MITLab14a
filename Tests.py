
import pickle
import numpy
import math
from wav_utils import *
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

def write_wav():
    # write a 2 second sound file of a 1khz tone
    seconds = 2
    sample_rate = 44100
    t = numpy.arange(sample_rate * 2)

    f = 1000

    def amp(x):
        time = (x / sample_rate )
        return numpy.cos(time * (2 * numpy.pi) * f)

    wav = amp(t)

    plt.plot(wav[:1000])
    plt.show()

    wav_write(wav, sample_rate, "out.wav")


def write_test():
    f = 1000
    omega = (610e+3) * 2 * numpy.pi

    dt = 1.0e-7
    rate = int(1/dt)
    seconds = 2
    t = numpy.arange(seconds * rate)

    def eval(x):
        time = x * dt
        signal = numpy.cos(time * (2 * numpy.pi) * f)
        carrier = numpy.cos(time * omega)
        mod = signal * carrier
        return mod

    signal = eval(t)

    plt.plot(signal[1:100000])
    plt.show()

    dtv = fft(signal)
    plt.plot(numpy.real(dtv))
    plt.show()

    with open('test_signal.pkl', 'wb') as f:
        pickle.dump(signal, f)


write_test()


    
