# RR_indices is the rr-distance between two peaks (peaklist): (peaklist[i] , peaklist[i+1])
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np


def FindTruePeaks(wd):
    peaks = wd['peaklist']
    binary = wd['binary_peaklist']

    result = peaks * binary
    result = [i for i in result if i != 0]

    return result


def CalculateHRVSignal(wd, visualize = False):
    HRV_signal = []
    # calculate HRV signal - nan before and after
    last = np.nan
    peaks = FindTruePeaks(wd)
    for i in range(0, peaks[0]):
        HRV_signal.append(np.nan)
    for i in range(len(peaks) - 1):
        new = peaks[i + 1] - peaks[i]
        for i in range(peaks[i], peaks[i + 1]):
            HRV_signal.append(np.abs(new - last))
        last = new
    for i in range(len(wd['hr']) - len(HRV_signal)):
        HRV_signal.append(np.nan)

    if (visualize):
        # set large figure
        plt.figure(figsize=(20, 4))
        plt.plot((HRV_signal))
        plt.title('HRV')

    return HRV_signal


def VisualizeSignalPeaks(signal, wd):
    plt.figure(figsize=(20, 4))
    plt.plot(signal)

    for xc in (FindTruePeaks(wd)):
        plt.axvline(x=xc, c='g')
    plt.show()


def AvgSignal(hrv, wd, step=10):
    AVG = []
    peaks = FindTruePeaks(wd)

    for i in range(0, len(peaks), step):
        if i == 0:
            start = 0
        else:
            start = peaks[i]

        if i + step < len(peaks):
            end = peaks[i + step]
        else:
            end = len(hrv)

        avg = np.nanmedian(hrv[start:end])

        for j in range(start, end):
            AVG.append(avg)

    return AVG

def ProcessSignalPPG(signal, visualize = False):
    # filter the signal
    filtered = hp.filter_signal(signal, cutoff=[0.75, 3], sample_rate=128, order=2,
                                filtertype='bandpass')  # [0.75, 3.5]
    # run the analysis
    wd, m = hp.process(filtered, sample_rate=128)

    if(visualize):
        VisualizeSignalPeaks(signal, wd)

    hrv = CalculateHRVSignal(wd, visualize)
    avg = AvgSignal(hrv, wd)
    return hrv, avg

