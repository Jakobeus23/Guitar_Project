import os
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Masking, GRU, TimeDistributed, Dense
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FOR out_of_range error default w/ batches || 0 = all logs, 1 = warnings, 2 = errors only, 3 = fatal errors

###################################
# Configuration / Hyper‑params
###################################
LABEL_CSV  = "labels.csv"
DATA_DIR   = "Guitar_Data"

FFT_SIZE   = 2048
HOP_LENGTH = 512
SAMPLE_RATE= 44100
MIN_FREQ   = 30.0
MAX_FREQ   = SAMPLE_RATE / 2
N_LOG_BINS = 512

TEST_SPLIT = 0.2
BATCH_SIZE = 8
EPOCHS      = 35
RNN_UNITS   = 64
LEARNING_RATE = 1e-3
DROPOUT     = 0.1

###################################
# 1. Read label file
###################################
df = pd.read_csv(LABEL_CSV)
unique_labels = sorted(df["label"].unique())
num_classes   = len(unique_labels)
label_to_idx  = {lbl:i for i,lbl in enumerate(unique_labels)}
idx_to_label  = {i:lbl for lbl,i in label_to_idx.items()}
print("Unique labels:", unique_labels)

###################################
# 2. Spectrogram helper
###################################

def compute_log_spectrogram(y, sr):
    S = np.abs(librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    log_bins = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), N_LOG_BINS)
    out = np.zeros((S_db.shape[1], N_LOG_BINS), dtype=np.float32)
    for k,f in enumerate(log_bins):
        idx = np.argmin(np.abs(freqs-f))
        out[:,k] = S_db[idx]
    # row‑wise z‑score (time axis) – stabilises gradients
    out = (out - out.mean(axis=1, keepdims=True)) / (out.std(axis=1, keepdims=True) + 1e-6)
    return out


###################################
# 3. tf.data loaders
###################################

def load_and_process(fname, label_vec):
    """
    Returns 3 things:
      spec         Float32 tensor, shape (T, N_LOG_BINS)
      frame_labels Float32 tensor, shape (T, num_classes)
      weights      Float32 tensor, shape (T,)
    """
    def _py_load(path_bytes, label_np):
        path = os.path.join(DATA_DIR, path_bytes.decode())
        y, sr = sf.read(path)

        # stereo → mono
        if y.ndim == 2:
            if y.shape[1] == 1:
                y = y[:,0]
            else:
                ch0, ch1 = y[:,0], y[:,1]
                y = ch0 if np.sum(np.abs(ch0))>np.sum(np.abs(ch1)) else ch1

        # compute spec
        spec = compute_log_spectrogram(y, sr).astype(np.float32)  # (T, bins)
        T    = spec.shape[0]

        # repeat the one‑hot per frame
        frame_labels = np.repeat(label_np[None, :], T, axis=0).astype(np.float32)
        # weight=1 for real, we'll pad weights later to 0
        weights = np.ones((T,), dtype=np.float32)

        return spec, frame_labels, weights

    spec, frame_labels, weights = tf.numpy_function(
        _py_load,
        [fname, label_vec],
        [tf.float32, tf.float32, tf.float32]
    )
    spec.set_shape([None, N_LOG_BINS])
    frame_labels.set_shape([None, num_classes])
    weights.set_shape([None])
    return spec, frame_labels, weights





# split file paths / labels
paths = df["filename"].tolist()
labels_idx = df["label"].map(label_to_idx).tolist()
train_p, val_p, train_l, val_l = train_test_split(paths, labels_idx, test_size=TEST_SPLIT,
                                                 stratify=labels_idx, random_state=42)

# assume you’ve already split:
#   train_p, val_p = lists of filenames
#   train_l, val_l = lists of label indices

def make_ds(paths, lbls, shuffle=False, repeat=False):
    onehot = tf.keras.utils.to_categorical(lbls, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((paths, onehot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            [None, N_LOG_BINS],    # spec
            [None, num_classes],   # frame_labels
            [None]                 # weights
        ),
        padding_values=(
            0.0,  # pad spec
            0.0,  # pad labels
            0.0   # pad weights (0 = ignored)
        )
    )
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


# usage
train_ds = make_ds(train_p, train_l, shuffle=True,  repeat=True)
val_ds   = make_ds(val_p,   val_l, shuffle=False, repeat=True)



train_ds = make_ds(train_p, train_l, shuffle=True,  repeat=True)
val_ds   = make_ds(val_p,   val_l, shuffle=False, repeat=True)


train_ds = make_ds(train_p, train_l, shuffle=True,  repeat=True)

# 1) See one batch
# Sanity‑check one training batch
# Sanity‑check one training batch
for spec_batch, frame_labels_batch, weights_batch in train_ds.take(1):
    # spec_batch  : (batch_size, time, bins)
    # frame_labels_batch : (batch_size, time, classes)
    # weights_batch : (batch_size, time)
    print("Spectrogram batch shape   :", spec_batch.shape)
    print("Frame‑labels batch shape  :", frame_labels_batch.shape)
    print("Weights batch shape       :", weights_batch.shape)

    # Show the first frame’s label for each sample
    for i in range(spec_batch.shape[0]):
        onehot  = frame_labels_batch[i, 0].numpy()
        decoded = idx_to_label[int(np.argmax(onehot))]
        print(f" Sample {i} first‑frame label → {decoded} (one‑hot {onehot})")
    break




val_ds = make_ds(val_p, val_l, shuffle=False, repeat=True)
validation_steps = max(1, len(val_p) // BATCH_SIZE)


steps_per_epoch  = math.ceil(len(train_p)/BATCH_SIZE)
validation_steps = max(1, math.ceil(len(val_p)/BATCH_SIZE))

###################################
# 4. Model – CNN (no pooling) + 2‑layer GRU
###################################
inp = Input(shape=(None, N_LOG_BINS), name="spec_input")   # (batch, time, bins)

# ---- two GRU layers, both returning sequences ----
x   = GRU(RNN_UNITS, return_sequences=True, name="gru1")(inp)
x   = GRU(RNN_UNITS, return_sequences=True, name="gru2")(x)

# ---- per‑frame dense + sigmoid (multilabel) ----
x   = TimeDistributed(Dense(64, activation="relu"), name="td_dense1")(x)
out = TimeDistributed(Dense(num_classes, activation="sigmoid"), name="td_output")(x)

model = Model(inputs=inp, outputs=out, name="seq_crnn")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

###################################
# 5. Train
###################################
model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS
)


###################################
# 6. Evaluate on validation set
###################################

loss, acc = model.evaluate(val_ds, steps=validation_steps)
print(f"Val loss: {loss:.4f} | Val acc: {acc:.4f}")

###################################
# 7. Prediction report
###################################
def load_with_name(fname, label_vec):
    # we only need the spectrogram + the original clip‑level one‑hot label + filename
    spec, frame_labels, weights = load_and_process(fname, label_vec)
    return spec, label_vec, fname


pred_ds = (
    tf.data.Dataset
      .from_tensor_slices((val_p, tf.keras.utils.to_categorical(val_l, num_classes)))
      .map(load_with_name, num_parallel_calls=tf.data.AUTOTUNE)
      .padded_batch(
          BATCH_SIZE,
          padded_shapes=(
            [None, N_LOG_BINS],   # spec: (time, bins)
            [num_classes],        # label: (classes,)
            []                    # fname: scalar
          )
      )
      .prefetch(tf.data.AUTOTUNE)
)


THR = 0.35
print("\nF U L L   P R E D I C T I O N   R E P O R T\n" + "=" * 43)
miss_rows = []

for specs, labels, fnames in pred_ds:
    seq_probs = model.predict(specs, verbose=0)      # -> (batch, time, classes)
    for j in range(seq_probs.shape[0]):
        fn          = fnames[j].numpy().decode()
        true_vec    = labels[j].numpy()               # (classes,)
        per_frame   = seq_probs[j]                    # (time, classes)
        clip_prob   = per_frame.mean(axis=0)          # now (classes,)

        # threshold‑or‑fallback
        pred_vecs = (clip_prob >= THR).astype(int)
        if not pred_vecs.any():
            top1 = clip_prob.argmax()
            pred_vecs[top1] = 1

        # decode labels
        true_lbls = [idx_to_label[k] for k in np.where(true_vec==1)[0]]
        pred_lbls = [idx_to_label[k] for k in np.where(pred_vecs==1)[0]]

        # format probs
        prob_str = ", ".join(
            f"{idx_to_label[k]}:{clip_prob[k]:.2f}"
            for k in range(len(clip_prob))
        )

        print(f"{fn:<25s} | TRUE: {', '.join(true_lbls):<15} → PRED: {', '.join(pred_lbls)}")
        print(f"  Probs: {prob_str}")

        # count misses by top‑1
        top1_pred = clip_prob.argmax()
        if np.argmax(true_vec) != top1_pred:
            miss_rows.append({
                "filename": fn,
                "true_labels": ";".join(true_lbls),
                "pred_labels": idx_to_label[top1_pred]
            })

print(f"\nTotal mis‑classified clips: {len(miss_rows)} / {len(val_p)}")
if miss_rows:
    pd.DataFrame(miss_rows).to_csv("misclassified_clips.csv", index=False)
    print("Saved details to misclassified_clips.csv")


###################################
# 8. Predict on a single file
###################################
AUDIO_FILE = "First_test (1).wav"  # path to your audio file

def compute_log_spectrogram_test(y, sr):
    S = np.abs(librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    log_bins = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), N_LOG_BINS)
    out = np.zeros((S_db.shape[1], N_LOG_BINS), dtype=np.float32)
    for k, f in enumerate(log_bins):
        idx = np.argmin(np.abs(freqs - f))
        out[:, k] = S_db[idx]
    out = (out - out.mean(axis=1, keepdims=True)) / (out.std(axis=1, keepdims=True) + 1e-6)
    return out

# ----------------------------
# Step 1: Load and preprocess audio
# ----------------------------
y, sr = sf.read(AUDIO_FILE)
if y.ndim == 2:  # stereo → mono
    y = y[:, 0] if np.sum(np.abs(y[:, 0])) > np.sum(np.abs(y[:, 1])) else y[:, 1]

spec = compute_log_spectrogram_test(y, sr)  # shape: (T, bins)
spec = np.expand_dims(spec, axis=0)    # add batch dim: (1, T, bins)


preds = model.predict(spec, verbose=0)[0]  # shape: (T, num_classes)

# ----------------------------
# Step 3: Aggregate predictions (clip-level)
# ----------------------------
predicted_seq = []

# For each frame, find the top predicted class (if above threshold)
for frame in preds:
    if frame.max() < THR:
        predicted_seq.append("silence")
    else:
        predicted_seq.append(idx_to_label[frame.argmax()])

# Collapse consecutive duplicates
ordered_labels = []
prev_label = None
for label in predicted_seq:
    if label != prev_label:
        ordered_labels.append(label)
        prev_label = label

# Remove silence from result (optional)
ordered_notes = [lbl for lbl in ordered_labels if lbl != "silence"]

print(f"Predicted Note Sequence: {ordered_notes}")

# Decode final predicted classes

# Optional: Frame-wise plot
plt.imshow(preds.T, aspect='auto', origin='lower', cmap='magma')
plt.colorbar()
plt.yticks(np.arange(len(idx_to_label)), [idx_to_label[i] for i in range(len(idx_to_label))])
plt.title("Per-frame Prediction Probabilities")
plt.xlabel("Time Frame")
plt.ylabel("Label")
plt.tight_layout()
plt.show()