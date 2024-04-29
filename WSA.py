from scipy.signal import resample
from scipy.signal import find_peaks
import numpy as np

def WSA(ecg_signal,fs,per):
    num_samples, num_leads, signal_length = ecg_signal.shape
    processed_batch = np.zeros((num_samples, num_leads, 1200))

    # Each set of signals is processed
    for i in range(num_samples):
        # Each 12-lead signal is processed
        for j in range(num_leads):
            # The signal is downsampled
            downsampled_signal = resample(ecg_signal[i, j], signal_length//per)
            rpeaks, rrs = find_peaks(downsampled_signal, distance=90)
            if len(rpeaks) >= 2:
                for k in range(len(rpeaks)):
                    beg_qrs = max(0, rpeaks[k] - int(0.05 * fs//per))
                    end_qrs = min(signal_length // per, rpeaks[k] + int(0.05 * fs//per))

                    scaling_factor = np.random.uniform(0.5, 1.5)  # The range of random factors can be adjusted to suit your needs
                    downsampled_signal[beg_qrs:end_qrs] *= scaling_factor
                    processed_batch[i, j, :] = downsampled_signal[:1200]

    return processed_batch