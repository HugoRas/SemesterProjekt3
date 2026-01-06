import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import wfdb
import os


data_dir = 'EKG datafiles/ecg-id-database-1.0.0/Person_01'

# optagelse
record_name = 'rec_1'


signals, info = wfdb.rdsamp(
    os.path.join(data_dir, record_name),
    channels=[0, 1],
    sampfrom=0,
    sampto=1000
)

# Opdel rå og filtreret signal
ecg0 = signals[:, 0]   # Rå ECG
ecg1 = signals[:, 1]   # Filtreret reference


fs = info['fs']

# Tidakse
t = np.arange(len(ecg0)) / fs



# Højpasfilter (fjerner baseline wander)
b_hp, a_hp = signal.butter(4, 0.5, btype='highpass', fs=fs)
ecg_hp = signal.filtfilt(b_hp, a_hp, ecg0)

# Notchfilter (50 Hz netstøj)
b_notch, a_notch = signal.iirnotch(50, 30, fs=fs)
ecg_hp_n = signal.filtfilt(b_notch, a_notch, ecg_hp)

# Lavpasfilter (fjerner højfrekvent støj)
b_lp, a_lp = signal.butter(4, 40, btype='lowpass', fs=fs)
ecg_filtered = signal.filtfilt(b_lp, a_lp, ecg_hp_n)

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(12, 8))

# Rå ECG
plt.subplot(2, 2, 1)
plt.plot(t, ecg0)
plt.title('Rå ECG')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude')

# Spektrum af rå ECG
plt.subplot(2, 2, 2)
plt.magnitude_spectrum(ecg0, Fs=fs)
plt.title('Spektrum – Rå ECG')

# Filtreret ECG vs reference
plt.subplot(2, 2, 3)
plt.plot(t, ecg_filtered, label='Filtreret (egen)')
plt.plot(t, ecg1, '--', label='Reference (database)')
plt.title('Filtreret ECG')
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude')
plt.legend()

# Spektrum af filtreret ECG
plt.subplot(2, 2, 4)
plt.magnitude_spectrum(ecg_filtered, Fs=fs)
plt.title('Spektrum – Filtreret ECG')

plt.tight_layout()
plt.show()
