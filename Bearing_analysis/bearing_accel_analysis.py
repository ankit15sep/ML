import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

'''
Set No. 1:
    Recording Duration: October 22, 2003 12:06:24 to November 25, 2003 23:39:56
    No. of Files: 2,156
    No. of Channels: 8
    Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4;
    Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8.
    File Recording Interval: Every 10 minutes (except the first 43 files were taken every 5 minutes)
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, inner race defect occurred in
    bearing 3 and roller element defect in bearing 4.
Set No. 2:
    Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39
    No. of Files: 984
    No. of Channels: 4
    Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
    File Recording Interval: Every 10 minutes
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, outer race failure occurred in
    bearing 1.
Set No. 3
    Recording Duration: March 4, 2004 09:27:46 to April 4, 2004 19:01:57
    No. of Files: 4,448
    No. of Channels: 4
    Channel Arrangement: Bearing1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing4 – Ch4;
    File Recording Interval: Every 10 minutes
    File Format: ASCII
    Description: At the end of the test-to-failure experiment, outer race failure occurred in
    bearing 3.
'''

def get_psd(data: np.ndarray, fs: float = 1000.0, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of the given data.

    Args:
        data (np.ndarray): Input signal data.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: PSD of the input data.
        :param nperseg: buffer length to use for the FFT.
    """
    f, pxx = signal.welch(data, fs=fs, window='hann', nperseg=nperseg)
    return f, pxx


df_original = pd.read_hdf('data/raw/combined_test1.h5', key='df')
df_resample = pd.read_hdf('data/raw/combined_test1_resample_1000Hz.h5', key='df')

t_original = np.arange(df_original.shape[0]) / 20000
t_resample = np.arange(df_resample.shape[0]) / 1000

plt.plot(t_original, df_original['bearing_1_x'].to_numpy())
plt.plot(t_resample, df_resample['bearing_1_x'].to_numpy())
plt.xlabel('Time (s)')
plt.ylabel('Bearing 1 X-axis Acceleration')
plt.title('Bearing 1 X-axis Acceleration Comparison')
plt.legend(['Original 20kHz', 'Resampled 1kHz'])
plt.show()

f_og, pxx_og = get_psd(df_original['bearing_1_x'].to_numpy(), fs=20000, nperseg=16384)
f_re, pxx_re = get_psd(df_resample['bearing_1_x'].to_numpy(), fs=1000, nperseg=1024)
plt.semilogy(f_og, pxx_og, label='Original 20kHz')
plt.semilogy(f_re, pxx_re, label='Resampled 1kHz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.title('Power Spectral Density Comparison')
plt.legend()
plt.show()
plt.show()
