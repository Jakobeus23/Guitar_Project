import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

###################################
# Configuration / Hyperparams
###################################
LABEL_CSV = "labels.csv"
DATA_FOLDER = "Guitar_Data"

FFT_SIZE = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050  # downsample for efficiency if needed
MIN_FREQ = 30.0
MAX_FREQ = SAMPLE_RATE / 2.0
N_LOG_BINS = 512

TEST_SPLIT = 0.2
BATCH_SIZE = 8
EPOCHS = 10
RNN_UNITS = 64
LEARNING_RATE = 1e-3

DROPOUT_RATE = 0.1

###################################
# 1. Load the CSV and Identify Labels
###################################
df = pd.read_csv(LABEL_CSV)
unique_labels = sorted(df["label"].unique())
num_classes = len(unique_labels)
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

print("Unique labels:", unique_labels)

###################################
# 2. Function to compute log-frequency STFT
###################################
def compute_log_spectrogram(audio, sr):
    """
    1) Pad short audio if needed
    2) STFT -> amplitude_to_db
    3) Map freq bins to log scale
    Returns shape (N_LOG_BINS, time_frames).
    """
    # If audio < FFT_SIZE, pad
    if len(audio) < FFT_SIZE:
        pad_len = FFT_SIZE - len(audio)
        audio = np.pad(audio, (0, pad_len))

    stft = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    log_freqs = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), num=N_LOG_BINS)

    time_frames = mag_db.shape[1]
    log_spectrogram = np.zeros((N_LOG_BINS, time_frames), dtype=np.float32)

    for i, lf in enumerate(log_freqs):
        idx = np.argmin(np.abs(freqs - lf))
        row = mag_db[idx, :].ravel()  # ensure shape (time_frames,)
        log_spectrogram[i, :] = row

    return log_spectrogram

###################################
# 3. Load/Preprocess Audio, Pad Spectrograms
###################################
X_spectrograms = []
y_labels = []
max_time_frames = 0

for _, row in df.iterrows():
    wav_file = row["filename"]
    label_str = row["label"]

    wav_path = os.path.join(DATA_FOLDER, wav_file)
    if not os.path.isfile(wav_path):
        print(f"Warning: {wav_path} not found, skipping.")
        continue

    # Load and downsample
    audio, sr = sf.read(wav_path)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Build multi-label vector (currently single label -> 1.0)
    label_vec = np.zeros(num_classes, dtype=np.float32)
    label_index = label_to_idx[label_str]
    label_vec[label_index] = 1.0

    # Compute log-spectrogram
    log_spec = compute_log_spectrogram(audio, sr)  # (N_LOG_BINS, time_frames)
    cur_frames = log_spec.shape[1]
    if cur_frames > max_time_frames:
        max_time_frames = cur_frames

    X_spectrograms.append(log_spec)
    y_labels.append(label_vec)

print("Total samples loaded:", len(X_spectrograms))
print("Max time frames:", max_time_frames)

# Pad all to max_time_frames
num_samples = len(X_spectrograms)
X_data = np.zeros((num_samples, N_LOG_BINS, max_time_frames, 1), dtype=np.float32)
y_data = np.array(y_labels, dtype=np.float32)

for i, spec in enumerate(X_spectrograms):
    t_frames = spec.shape[1]
    X_data[i, :, :t_frames, 0] = spec  # zero-pad the remainder

print("X_data shape:", X_data.shape)
print("y_data shape:", y_data.shape)

###################################
# 4. Train / Test Split
###################################
indices = np.arange(num_samples)
np.random.shuffle(indices)
split_idx = int(num_samples * (1 - TEST_SPLIT))

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train = X_data[train_idx]
y_train = y_data[train_idx]
X_test = X_data[test_idx]
y_test = y_data[test_idx]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

###################################
# 5. Build the Model (No Pooling, With Dropout)
###################################
freq_bins = N_LOG_BINS
time_dim = max_time_frames

inputs = layers.Input(shape=(freq_bins, time_dim, 1))

# ---- Convolutional layers, NO POOLING, but with dropout ----
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
x = layers.Dropout(DROPOUT_RATE)(x)

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.Dropout(DROPOUT_RATE)(x)
# shape: (None, freq_bins, time_dim, 32)

# ---- Now reshape for RNN ----
shape_after_cnn = tf.keras.backend.int_shape(x)
_, freq_post, time_post, chan_post = shape_after_cnn
feat_dim = freq_post * chan_post  # combine freq and channels

# We assume each time frame is the second dimension we want for RNN
# so we permute: (batch, freq, time, chan) -> (batch, time, freq, chan)
x = layers.Permute((2,1,3))(x)  # (None, time_dim, freq_bins, 32)
x = layers.Reshape((time_post, feat_dim))(x)  # (None, time_dim, freq_bins*32)

# ---- RNN layer with dropout ----
x = layers.GRU(RNN_UNITS, return_sequences=False)(x)
x = layers.Dropout(DROPOUT_RATE)(x)

# ---- Dense -> Sigmoid for multi-label ----
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

###################################
# 6. Training
###################################
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

###################################
# 7. Evaluation
###################################
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


import numpy as np
import pandas as pd

# -----------------------------------------------
#  AFTER  model.evaluate(...) add this block
# -----------------------------------------------

# threshold for sigmoid → binary
THR = 0.35

# 1) run inference on the test set
y_prob = model.predict(X_test, batch_size=8)

label_to_idx = {'5_0': 0, '6_0': 1, '6_5': 2, 'silence': 3}

# --- inspect predictions before thresholding ---
print("\nF U L L   P R E D I C T I O N   R E P O R T")
print("===========================================\n")

for i, (true_vec, prob_vec) in enumerate(zip(y_test, y_prob)):
    test_sample_idx = test_idx[i]
    filename = df.iloc[test_sample_idx]['filename']

    idx_to_label = {i: label for label, i in label_to_idx.items()}

    true_labels = [idx_to_label[j] for j in np.where(true_vec == 1)[0]]

    # Predicted label(s)
    pred_vec = (prob_vec >= THR).astype(int)
    pred_labels = [idx_to_label[j] for j in np.where(pred_vec == 1)[0]]

    # Format probability output
    prob_rounded = {idx_to_label[j]: float(f"{prob_vec[j]:.3f}") for j in range(len(prob_vec))}

    # Display all results
    print(f"{filename:20s} | TRUE: {', '.join(true_labels):<12} → PRED: {', '.join(pred_labels)}")
    print(f"  Probs: {prob_rounded}\n")


# Always predict the class with the highest probability
y_pred = np.zeros_like(y_prob)
top1 = np.argmax(y_prob, axis=1)
y_pred[np.arange(len(y_prob)), top1] = 1


# 2) utility: map index ↔ label
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

# 3) list to collect misses
miss_rows = []

print("\nM I S S  C A S E S")
print("--------------------")
for i, (true_vec, pred_vec) in enumerate(zip(y_test, y_pred)):
    if not np.array_equal(true_vec, pred_vec):
        # retrieve filename of this sample
        test_sample_idx = test_idx[i]      # original row in full dataset
        wav_file = df.iloc[test_sample_idx]["filename"]

        # decode the multi‑hot vectors to label lists
        true_labels = [idx_to_label[j] for j in np.where(true_vec == 1)[0]]
        pred_labels = [idx_to_label[j] for j in np.where(pred_vec == 1)[0]]

        print(f"{wav_file:20s} | true: {true_labels}  →  pred: {pred_labels}")

        miss_rows.append({
            "filename": wav_file,
            "true_labels": ";".join(true_labels),
            "pred_labels": ";".join(pred_labels)
        })

print(f"\nTotal mis‑classified clips: {len(miss_rows)} / {len(X_test)}")

# 4) save to CSV (optional)
if miss_rows:
    pd.DataFrame(miss_rows).to_csv("misclassified_clips.csv", index=False)
    print("Saved details to misclassified_clips.csv")
