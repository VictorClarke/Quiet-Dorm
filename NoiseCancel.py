__author__ = 'Victor'
import sys
import math
import wave
import struct
import curses
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

standard = curses.initscr()
standard.nodelay(True)
curses.noecho()
curses.cbreak()

pythonAudioObject = pyaudio.PyAudio()
MODE = sys.argv[1]
FOLD = 1
SAMPLE_RATE = 44100
CHANNELS = 2
WIDTH = 2


try:
    IterationsN = int(sys.argv[3])
except (ValueError, IndexError):
    print('The second argument has to be a number.')
    sys.exit()


def main():
    if MODE == '--live':
        live_mode()
    else:
        print('Please write live-mode with the first argument')


def live_mode():
    standard.addstr('Noise-cancelling live')

    stream = pythonAudioObject.open(
        format=pythonAudioObject.get_format_from_width(WIDTH),
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=FOLD,
        input=True,
        output=True
    )

    decibel_levels = []
    total_original = []
    total_inverted = []
    total_difference = []

    active = True

    # Fold the data apart in iterations according to the preset constants
    try:
        for i in range(0, int(SAMPLE_RATE / FOLD * sys.maxunicode)):
            pressed_key = standard.getch()
            # Press the 'a' key to toggle active
            if pressed_key == 97:
                active = not active
            # If the 'q' key was pressed quit the constant loop
            if pressed_key == 113:
                break

            # Read in a fold of live audio for each iteration
            original = stream.read(FOLD)

            inverted = invert(original)
            stream.write(inverted, FOLD)

            # On every nth iteration append the difference between the level of the source audio and the inverted one
            if i % IterationsN == 0:
                standard.clear()
                difference = calculate_difference(original, inverted)
                standard.addstr('Difference (in dB): {}'.format(difference))
                decibel_levels.append(difference)
                int_original, int_inverted, int_difference = calculate_wave(original, inverted)
                total_original.append(int_original)
                total_inverted.append(int_inverted)
                total_difference.append(int_difference)

    except (KeyboardInterrupt, SystemExit):
        print('Noise cancelling has been terminated')
        if sys.argv[2] == '--decibel' or sys.argv[2] == '-db':
            plot_results(decibel_levels, IterationsN)
        elif sys.argv[2] == '--waves' or sys.argv[2] == '-wv':
            plot_wave_results(total_original, total_inverted, total_difference, IterationsN)

        curses.endwin()
        stream.stop_stream()
        stream.close()
        pythonAudioObject.terminate()
        sys.exit()


def invert(data):
    """
    Inverts the byte data it received utilizing an XOR operation.
    :param data: A chunk of byte data
    :return inverted: The same size of chunked data inverted bitwise
    """
    intwave = np.fromstring(data, np.int32)
    intwave = np.invert(intwave)
    inverted = np.frombuffer(intwave, np.byte)
    return inverted


def mix_samples(sample_1, sample_2, ratio):
    """
    Mixes two samples into each other
    :param sample_1: A bytestring containing the first audio source
    :param sample_2: A bytestring containing the second audio source
    :param ratio: A float which determines the mix-ratio of the two samples (the higher, the louder the first sample)
    :return mix: A bytestring containing the two samples mixed together
    """
    (ratio_1, ratio_2) = get_ratios(ratio)
    intwave_sample_1 = np.fromstring(sample_1, np.int16)
    intwave_sample_2 = np.fromstring(sample_2, np.int16)
    intwave_mix = (intwave_sample_1 * ratio_1 + intwave_sample_2 * ratio_2).astype(np.int16)
    mix = np.frombuffer(intwave_mix, np.byte)
    return mix


def get_ratios(ratio):
    """
    Calculates the ratios using a received float
    :param ratio: A float betwenn 0 and 2 resembling the ratio between two things
    :return ratio_1, ratio_2: The two calculated actual ratios
    """
    ratio = float(ratio)
    ratio_1 = ratio / 2
    ratio_2 = (2 - ratio) / 2
    return ratio_1, ratio_2


def calculate_decibel(data):
    """
    Calculates the volume level in decibel of the given data
    :param data: A bytestring used to calculate the decibel level
    :return db: The calculated volume level in decibel
    """
    count = len(data) / 2
    form = "%dh" % count
    shorts = struct.unpack(form, data)
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0 / 32768)
        sum_squares += n * n
    rms = math.sqrt(sum_squares / count) + 0.0001
    db = 20 * math.log10(rms)
    return db


def calculate_difference(data1, data2):
    """
    Calculates the difference level in decibel between the received binary inputs
    :param data1: The first binary digit
    :param data2: The second binary digit
    :return difference: The calculated difference level (in dB)
    """
    difference = calculate_decibel(data1) - calculate_decibel(data2)
    return difference


def calculate_wave(original, inverted, ratio):
    """
    Converts the bytestrings it receives into plottable integers and calculates the difference between both
    :param original: A bytestring of sound
    :param inverted: A bytestring of sound
    :param ratio: A float which determines the mix-ratio of the two samples
    :return originalInt, invertedInt, differenceInt: A tupel of the three calculated integers
    """
    (ratio1, ratio2) = get_ratios(ratio)
    originalInt = np.fromstring(original, np.int16)[0] * ratio1
    invertedInt = np.fromstring(inverted, np.int16)[0] * ratio2
    differenceInt = (originalInt + invertedInt)
    return originalInt, invertedInt, differenceInt


def plot_results(data, nth_iteration):
    """
    Plots the list it receives and cuts off the first ten entries to circumvent the plotting of initial silence
    :param data: A list of data to be plotted
    :param nth_iteration: Used for the label of the x axis
    """
    plt.plot(data[10:])
    plt.xlabel('Time (every {}th {} byte)'.format(nth_iteration, FOLD))
    plt.ylabel('Volume level difference (in dB)')
    plt.suptitle('Difference - Median (in dB): {}'.format(np.round(np.fabs(np.median(data)), decimals=5)), fontsize=14)
    plt.show()


def plot_wave_results(total_original, total_inverted, total_difference, nth_iteration):
    """
    Plots the three waves of the original sound, the inverted one and their difference
    :param total_original: A list of the original wave data
    :param total_inverted: A list of the inverted wave data
    :param total_difference: A list of the difference of 'total_original' and 'total_inverted'
    :param nth_iteration: Used for the label of the x axis
    """
    plt.plot(total_original, 'b')
    plt.plot(total_inverted, 'r')
    plt.plot(total_difference, 'g')
    plt.xlabel('Time (per {}th {} byte chunk)'.format(nth_iteration, FOLD))
    plt.ylabel('Amplitude (integer representation of each {} byte chunk)'.format(nth_iteration, FOLD))
    plt.suptitle('Waves: original (blue), inverted (red), output (green)', fontsize=14)
    plt.show()


main()
