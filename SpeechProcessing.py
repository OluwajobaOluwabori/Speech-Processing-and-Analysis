from scipy.io.wavfile import read as read_wav
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram,welch
from scipy.signal.windows import hamming, hann, boxcar
from scipy.signal import find_peaks

#from scipy.special import title

#Load the recorded speech using the 'read_wav' function.
sampling_frequency, data = read_wav("C:\\Users\\user\\dev\\Speech Processing\\Recording.wav")
print(data)
print(data.shape)
print(sampling_frequency)
print(f"Number of channels = {data.shape[1]}")

##Basic Signal Characteristics:

#1a. Duration of the signal (in seconds).
length = data.shape[0] / sampling_frequency #total time
print(f"Duration of the signal (in seconds)= {length}s")


#1b. Sampling frequency and number of samples.
no_of_samples = data.shape[0]
print(f"Sampling frequency={sampling_frequency/1000} kHz and number of samples = {no_of_samples}")



signal = data[:,0]
# 1c. Mean value (average amplitude) and variance.
mean_signal = np.mean(signal)
variance = np.var(signal)
print(f'The average signal value is : {mean_signal:.2f} and variance of the signal is : {variance:.2f}')

#1d.Energy of the signal (sum of the squared values)
energy = np.sum(signal**2)
print(f'The energy of the signal is : {energy:.2f}')



#Plotting the speech signal
time = np.linspace(0,length,no_of_samples)
plt.figure(figsize=(12, 6))
plt.plot(time,data[:,0])
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title('Plot of the speech signal')
plt.show()

##Frequency Representation (Periodogram):

# Compute and plot the periodogram with different windowing functions
# def plot_periodogram(signal, fs, window_func, window_name):
#     # f contains the frequency components
#     # S is the PSD
#     f, S = periodogram(signal, fs, window=window_func, scaling='density')
#     plt.plot(f, 10 * np.log10(S), label=f'Window: {window_name}')
#
# plt.figure(figsize=(12, 6))
# # No window (rectangular)
# plot_periodogram(signal, sampling_frequency, 'boxcar', 'Rectangular')
# # Hamming window
# plot_periodogram(signal, sampling_frequency, hamming(len(signal)), 'Hamming')
# # Hanning window
# plot_periodogram(signal, sampling_frequency, hann(len(signal)), 'Hanning')
#
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power/Frequency (dB/Hz)')
# plt.title('Periodogram of Speech Signal with Different Windowing Functions')
# plt.legend()
# plt.grid()
# plt.show()



# Compute and plot the periodogram with different windowing functions
def plot_periodogram(asignal, fs, window_func):
    f, Pxx = periodogram(asignal, fs, window=window_func, scaling='density')
    return f, 10 * np.log10(Pxx)


# Create subplots for different window lengths
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 10))

# Original signal with full length
f_full, Pxx_full = plot_periodogram(signal, sampling_frequency, boxcar(len(signal)))

# Quarter length window
window_length_quarter = int(len(signal) / 4)
f_quarter, Pxx_quarter = plot_periodogram(signal[:window_length_quarter], sampling_frequency, boxcar(window_length_quarter))

# One-tenth length window
window_length_tenth = int(len(signal) / 100)
f_tenth, Pxx_tenth = plot_periodogram(signal[:window_length_tenth], sampling_frequency, boxcar(window_length_tenth))

# Plot each periodogram
axs[0].plot(f_full, Pxx_full, label='Window Length: 931,200', color='chocolate')
axs[1].plot(f_quarter, Pxx_quarter, label='Window Length: 232,800', color='green')
axs[2].plot(f_tenth, Pxx_tenth, label='Window Length: 9,312')

# Setting labels and titles for each subplot
for ax in axs:
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.grid(True)

axs[0].set_title('Window Length: 931,200', loc='left')
axs[1].set_title('Window Length: 232,800', loc='left')
axs[2].set_title('Window Length: 9,312', loc='left')
axs[2].set_xlabel('Frequency (Hz)')  # Set x-label for the last subplot

# Main title for the figure
fig.suptitle('Periodogram of Speech Signal with Different Window Lengths', fontsize=15)

# Show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()


#plt.figure(figsize=(12, 12))
# No window (rectangular)
fig, axs = plt.subplots(3,1, sharex=True,  figsize=(12, 10))
#
f,y = plot_periodogram(signal, sampling_frequency, 'boxcar')
# Hamming window
g,k = plot_periodogram(signal[:int(len(signal)/4)], sampling_frequency, boxcar(int(len(signal)/4)))
#g,k = plot_periodogram(signal, sampling_frequency, hamming(len(signal)), 'Hamming')
# Hanning window
d,l = plot_periodogram(signal[:int(len(signal)/100)], sampling_frequency, boxcar(int(len(signal)/100)))
axs[0].plot(f,y, label='931,200',color='chocolate')
axs[1].plot(g,k, label='232,800',color='green')
axs[2].plot(d,l, label='9,312')

# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power/Frequency (dB/Hz)')
# plt.title('Periodogram of Speech Signal with Different Windowing Functions')
# plt.legend()
# plt.grid()
# plt.show()
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])

axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power/Frequency (dB/Hz)')
axs[0].set_title('Window length:931200',loc = 'left')
axs[0].grid(True)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Power/Frequency (dB/Hz)')
axs[1].grid(True)
axs[1].set_title('Window length:232,800',loc = 'left')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Power/Frequency (dB/Hz)')
axs[2].set_title('Window length:9,312',loc = 'left')
axs[2].grid(True)
fig.suptitle('Periodogram of Speech Signal with Different Window Lengths', fontsize=15)
plt.show()


def find_and_plot_dominant_frequencies(frequencies, psd, height_threshold=25):
    """
    Find and plot dominant frequencies from a Power Spectral Density (PSD).

    Parameters:
    frequencies (array): Array of frequency values.
    psd (array): Array of Power Spectral Density values (in dB).
    height_threshold (float): Height threshold for peak detection (default is 25 dB).

    Returns:
    tuple: Dominant frequencies and their corresponding powers.
    """
    # Find peaks with the specified height threshold
    peaks, _ = find_peaks(psd, height=height_threshold)

    # Extract dominant frequencies and their corresponding power levels
    dominant_frequencies = frequencies[peaks]
    peak_powers = psd[peaks]
    # Plot the PSD and mark the dominant frequencies
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd, label='Power Spectral Density')
    plt.scatter(dominant_frequencies, peak_powers, color='red', label='Dominant Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Periodogram with Dominant Frequencies Marked')
    plt.legend()
    plt.grid(True)
    plt.show()

    return dominant_frequencies, peak_powers
dominant_freqs, peak_powers = find_and_plot_dominant_frequencies(f, y)


# Find peaks with a height threshold
# Assuming `f` and `Pxx_dB` are the frequency and PSD (in dB) from one of your windowed periodograms
peaks, _ = find_peaks(y, height=25)  # Adjust 'height' as necessary for sensitivity
dominant_frequencies = f[peaks]
peak_powers = y[peaks]

# Print dominant frequencies and their corresponding powers
print("Dominant Frequencies (Hz):", dominant_frequencies)
print("Peak Powers (dB):", peak_powers)
# Find the fundamental frequency as the lowest dominant frequency
fundamental_frequency = np.min(dominant_frequencies)

print("Fundamental Frequency (Hz):", fundamental_frequency)

plt.plot(f, y)
plt.scatter(dominant_frequencies, peak_powers, color='red', label='Dominant Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Periodogram with Dominant Frequencies Marked')
plt.legend()
plt.grid(True)
plt.show()



#different fs
from scipy.signal import resample

# Resample signal to different sampling frequencies
def plot_resampled_periodogram(signal,new_fs='new_fs'):
    """
        Resamples the signal to a new sampling frequency and computes the periodogram.

        Parameters:
        - signal: The input signal to resample and analyze.
        - new_fs: The new sampling frequency for resampling.

        Returns:
        - frequencies: Frequency components of the periodogram.
        - PSD: Power spectral density of the resampled signal.
        """
    # Resample the signal to the new sampling frequency
    # The original sampling frequency of the signal is 48kHZ
    resampled_signal = resample(signal, int(len(signal) * new_fs / sampling_frequency))

    # Compute the periodogram for the resampled signal
    frequencies, psd = plot_periodogram(resampled_signal, new_fs, 'boxcar', 'Rectangular')
    return  frequencies, psd

# Define sampling frequencies to analyze
sampling_frequencies = [52000, 48000, 16000]

# Define a dictionary to store results for easy retrieval
periodogram_data = {}

# Compute periodograms for each sampling frequency
for new_fs in sampling_frequencies:
    frequencies, psd = plot_resampled_periodogram(signal,  new_fs)
    periodogram_data[new_fs] = (frequencies, psd)
# Plot the periodograms for each resampling rate
for new_fs, (frequencies, psd) in periodogram_data.items():
    label = f"Resampled audio signal, fs={new_fs//1000}kHz" if new_fs != sampling_frequency else f"Original audio signal, fs={new_fs//1000}kHz"
    plt.plot(frequencies, psd, label=label)

# Customize plot

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Periodogram of Resampled Speech Signal')
plt.legend()
plt.grid()
plt.show()
