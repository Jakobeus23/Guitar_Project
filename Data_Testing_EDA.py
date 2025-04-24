import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_path = r"C:\Users\theja\OneDrive\Documents\Guitar_Project\Guitar_Project\Guitar_Data\6_0_1.wav"
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
# B) Linear-Scale Spectrogram (STFT)
########################################
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure()
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Linear-Scale Spectrogram")
plt.tight_layout()
plt.show()

########################################
# C) Constant-Q Transform (CQT)
########################################
C = librosa.cqt(y, sr=sr)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

plt.figure()
librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title("Constant-Q Transform (CQT)")
plt.tight_layout()
plt.show()

########################################
# D) Mel Spectrogram
########################################
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure()
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()

########################################
# E) Custom Log-Frequency STFT (used in your model)
########################################
FFT_SIZE   = 2048
HOP_LENGTH = 512
N_LOG_BINS = 512
MIN_FREQ   = 30.0
MAX_FREQ   = sr / 2

# STFT magnitude in dB
S = np.abs(librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Create log-spaced frequency bins
freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
log_bins = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), N_LOG_BINS)

# Interpolate onto log-frequency scale
log_spec = np.zeros((S_db.shape[1], N_LOG_BINS), dtype=np.float32)
for k, f in enumerate(log_bins):
    idx = np.argmin(np.abs(freqs - f))
    log_spec[:, k] = S_db[idx]

# Normalize (z-score) each time frame
log_spec = (log_spec - log_spec.mean(axis=1, keepdims=True)) / (log_spec.std(axis=1, keepdims=True) + 1e-6)

# Plot
plt.figure()
plt.imshow(log_spec.T, aspect='auto', origin='lower', cmap='magma',
           extent=[0, len(y)/sr, 0, N_LOG_BINS])
plt.colorbar(format='%+2.0f')
plt.title("Custom Log-Frequency STFT (Model Input)")
plt.xlabel("Time (s)")
plt.ylabel("Log-Frequency Bin")
plt.tight_layout()
plt.show()
