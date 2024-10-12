import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to plot the spectrogram of an audio file
def plot_spectrogram(time_series, sampling_rate, savepath):
    # Short-time Fourier transform
    fourier = librosa.stft(time_series) 
    # Convert the amplitude to decibels
    spectrogram_db = librosa.amplitude_to_db(np.abs(fourier), ref=np.max)
    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=time_series, sr=sampling_rate)
    times = librosa.times_like(spectral_centroid)

    # Plot the spectrogram with the spectral centroid
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Spectrogram')
    ax.plot(times, spectral_centroid.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.savefig(savepath)

# Path to the audio file
guitar_audio = 'all-samples/guitar/guitar_C4_very-long_forte_normal.mp3'
flute_audio = 'all-samples/flute/flute_C4_1_forte_normal.mp3'
saxophone_audio = 'all-samples/saxophone/saxophone_C4_1_forte_normal.mp3'
mandolin_audio = 'all-samples/mandolin/mandolin_C4_very-long_piano_normal.mp3'

# Paths to save the spectrograms
guitar_savepath = 'spectrograms/guitar_spectrogram.png'
flute_savepath = 'spectrograms/flute_spectrogram.png'
saxophone_savepath = 'spectrograms/saxophone_spectrogram.png'
mandolin_savepath = 'spectrograms/mandolin_spectrogram.png'

# load the audio files
guitar, sr_guitar = librosa.load(guitar_audio, offset=0.17, duration=1.2)
flute, sr_flute = librosa.load(flute_audio, duration=1.1)
saxophone, sr_saxophone = librosa.load(saxophone_audio, offset=0.02, duration=0.65)
mandolin, sr_mandolin = librosa.load(mandolin_audio, offset=0.33, duration=1.2)

# Plot the spectrograms
plot_spectrogram(guitar, sr_guitar, guitar_savepath)
plot_spectrogram(flute, sr_flute, flute_savepath)
plot_spectrogram(saxophone, sr_saxophone, saxophone_savepath)
plot_spectrogram(mandolin, sr_mandolin, mandolin_savepath)

# Fundamental frequency of the audio files
'''
f0_guitar = librosa.yin(guitar, sr=sr_guitar, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_flute = librosa.yin(flute, sr=sr_flute, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_saxophone = librosa.yin(saxophone, sr=sr_saxophone, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_violin = librosa.yin(violin, sr=sr_violin, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
'''

