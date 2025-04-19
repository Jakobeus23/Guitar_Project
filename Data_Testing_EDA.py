import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Load the audio file (replace with your actual path)
audio_path = r"C:\Users\theja\OneDrive\Documents\Guitar_Project\Guitar_Data\6_0_1.wav"
y, sr = librosa.load(audio_path, sr=None)

########################################
# A) Plot the Raw Waveform
########################################
plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

########################################
# B) Compute and Plot a Linear-Scale Spectrogram (STFT)
########################################
# Take the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)              # shape: (frequency_bins, time_frames)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  

# Plot the dB-scaled spectrogram
plt.figure()
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')  
plt.colorbar(format='%+2.0f dB')
plt.title("Linear-Scale Spectrogram")
plt.tight_layout()
plt.show()

########################################
# C) Another Representation (Optional) - e.g., Constant-Q Transform (CQT)
########################################
# If you want a CQT (log-frequency, musically related bins):
C = librosa.cqt(y, sr=sr)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

plt.figure()
# Use 'y_axis="cqt_note"' to label frequencies in musical notes
librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title("Constant-Q Transform (CQT)")
plt.tight_layout()
plt.show()

########################################
# (Optional) Mel Spectrogram
########################################
# If you still want to see a Mel spectrogram for comparison:
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure()
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()
