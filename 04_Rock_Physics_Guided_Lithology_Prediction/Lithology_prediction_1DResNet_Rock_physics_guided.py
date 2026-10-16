# -*- coding: utf-8 -*-

"""
Lithology_prediction_1DResNet_Rock_physics_guided.py
1DResNet-based lithology prediction using rock-physics-guided features.

Input features: PI, porosity, quartz, calcite, and clay contents.
Lithology classes: 1 = sandstone, 2 = mudstone,
                   3 = calcareous sandstone.

Training: shallowest 80% of depth samples.
Validation: deepest continuous 20% of depth samples.

The model uses a 1DResNet with PS-CALR learning-rate scheduling.
Full-well lithology predictions are generated after training.
"""


from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
# Must be set before importing TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random


# ============================================================
# 1. General settings
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
    'axes.unicode_minus': False,
})

# SEED = 2026
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
SEED = 2026
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

CSV_NAME = 'Caprock sensitive parameters and mineral contents.csv'
OUTPUT_DIR = 'Lithology_1DResNet_PS_CALR_Rock_physics_guided_results'

EPOCHS = 600                 # Original 1DResNet maximum epochs
BATCH_SIZE = 128             # Actual model.fit batch size
TEST_SIZE = 0.20
EARLY_STOP_PATIENCE = 200    # Original 1DResNet patience

# Exactly the step-guided PS-CALR settings from the second uploaded script
MAX_LR = 0.1
MIN_LR = 1e-2
LR_PERIOD = 1200            # optimizer steps, not epochs
LR_WARMUP = 10            # optimizer steps, not epochs

# Lith: 1 sandstone (yellow), 2 mudstone (black),
#       3 calcareous sandstone (blue).
LITH_COLORS = ['yellow', 'black', 'blue']
LITH_XLIM = (0, 10)
DEPTH_PAD_RATIO = 0      # 5% white space above/below the well section

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / OUTPUT_DIR
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Residual blocks retained from the original 1DResNet
# ============================================================
class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_channels, padding='same', kernel_size=3,
            strides=strides, kernel_regularizer='l2')
        self.conv2 = tf.keras.layers.Conv1D(
            num_channels, kernel_size=3, padding='same',
            kernel_regularizer='l2')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv1D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X, training=None):
        Y = tf.keras.activations.relu(
            self.bn1(self.conv1(X), training=training))
        Y = self.bn2(self.conv2(Y), training=training)
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super().__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=1))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X, training=None):
        for layer in self.residual_layers:
            X = layer(X, training=training)
        return X


# ============================================================
# 3. Load the single well CSV
# ============================================================
csv_candidates = [script_dir / CSV_NAME, Path.cwd() / CSV_NAME]
csv_path = next((p for p in csv_candidates if p.is_file()), None)
if csv_path is None:
    raise FileNotFoundError(
        f'Cannot find "{CSV_NAME}" beside the script or in the working directory. '
        'Please put the CSV there before running.')

well = pd.read_csv(csv_path, encoding='utf-8-sig')
well.columns = well.columns.str.strip()
feature_cols = ['True PI', 'True Por', 'True Vquar', 'True Vclc', 'True Vclay']
required_cols = ['Depth', *feature_cols, 'Lith']
missing_cols = [c for c in required_cols if c not in well.columns]
if missing_cols:
    raise ValueError(
        f'Missing CSV columns: {missing_cols}. Expected: {required_cols}')

well = well[required_cols].copy()
for col in required_cols:
    well[col] = pd.to_numeric(well[col], errors='coerce')

n_before = len(well)
well = well.dropna(subset=required_cols).sort_values('Depth').reset_index(drop=True)
if len(well) < n_before:
    print(f'Warning: excluded {n_before - len(well)} rows with missing/invalid values.')
if len(well) < 15:
    raise ValueError('At least 15 valid depth samples are required for this training setup.')

labels_raw = well['Lith'].to_numpy(dtype=np.float64)
if not np.all(np.isin(labels_raw, [1.0, 2.0, 3.0])):
    invalid = np.unique(labels_raw[~np.isin(labels_raw, [1.0, 2.0, 3.0])])
    raise ValueError(f'Lith must be coded as 1, 2 or 3, but found: {invalid}.')

Depth = well['Depth'].to_numpy(dtype=np.float64)
if np.any(np.diff(Depth) <= 0):
    raise ValueError('Depth must contain unique, increasing values for the strip plot.')

X_all_raw = well[feature_cols].to_numpy(dtype=np.float32)
y_all = labels_raw.astype(np.int32) - 1  # Keras labels: 0, 1, 2
n_classes = 3

print('Input CSV:', csv_path)
print('Feature columns:', feature_cols)
print('Samples:', len(Depth))
for class_id in (1, 2, 3):
    print(f'Lith class {class_id}: {(labels_raw == class_id).sum()} samples')

# Continuous depth holdout: the deepest 20% of depth-ordered samples.
# Depth has already been sorted in ascending order above.
n_val = int(np.ceil(TEST_SIZE * len(y_all)))
n_train = len(y_all) - n_val
train_idx = np.arange(0, n_train)
test_idx = np.arange(n_train, len(y_all))
if len(np.unique(y_all[train_idx])) < 2:
    raise ValueError('Training depth interval contains fewer than two lithology classes.')
print(f'Continuous-depth holdout: {n_train} training and {n_val} validation samples')
print(f'Training depth range: {Depth[train_idx[0]]:.4f}–{Depth[train_idx[-1]]:.4f} m')
print(f'Validation depth range: {Depth[test_idx[0]]:.4f}–{Depth[test_idx[-1]]:.4f} m')

# Fit scaler on TRAIN data only to avoid validation leakage.
scaler = StandardScaler().fit(X_all_raw[train_idx])
X_all = scaler.transform(X_all_raw).astype(np.float32)
X_train = np.expand_dims(X_all[train_idx], axis=2)  # [N, 5, 1]
X_test = np.expand_dims(X_all[test_idx], axis=2)
X_full = np.expand_dims(X_all, axis=2)
y_train = y_all[train_idx]
y_test = y_all[test_idx]


# ============================================================
# 4. One-dimensional ResNet classifier (original architecture)
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
        64, kernel_size=3, strides=1, padding='same',
        input_shape=(5, 1), kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    ResnetBlock(64, 1, first_block=True),  # Residual block 1
    tf.keras.layers.Dropout(0.5),
    ResnetBlock(128, 1),  # Residual block 2; 1x1 projection
    tf.keras.layers.Dropout(0.5),
    ResnetBlock(128, 1, first_block=True),  # Residual block 3
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=3)  # logits for 3 classes
])
model.summary()


# ============================================================
# 5. PS-CALR (warm-up + cosine annealing), from second script
# ============================================================
class PS_CALR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, min_lr, period, warmup):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.period = period
        self.warmup = warmup

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup
        cycle_step = tf.maximum(step - self.warmup, 0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + tf.cos(np.pi * cycle_step / self.period)
        )
        return tf.where(step < self.warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'period': self.period,
            'warmup': self.warmup,
        }


lr = PS_CALR(
    max_lr=MAX_LR,
    min_lr=MIN_LR,
    period=LR_PERIOD,
    warmup=LR_WARMUP,
)
# optimizer = tf.keras.optimizers.RMSprop(
#     learning_rate=lr, clipnorm=1.0)
optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


class CustomCallback(Callback):
    """Display current learning rate; does NOT modify the schedule."""
    def on_epoch_end(self, epoch, logs=None):
        current_step = self.model.optimizer.iterations
        current_lr = self.model.optimizer.learning_rate
        if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = current_lr(current_step)
        try:
            current_lr = float(tf.keras.backend.get_value(current_lr))
        except Exception:
            current_lr = float(lr(current_step).numpy())
        print(f'Epoch {epoch + 1}: optimizer steps={int(current_step.numpy())}, '
              f'learning rate={current_lr:.10f}')


earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=EARLY_STOP_PATIENCE,
    verbose=1, mode='min', restore_best_weights=True)
checkpoint_path = output_dir / 'best_1DResNet_PS_CALR.weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor='val_loss', mode='min',
    save_best_only=True, save_weights_only=True, verbose=0,
)

# ============================================================
# 6. Training
# ============================================================
time_start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[earlystopper, checkpoint, CustomCallback()],
    verbose=1,
)
print('Training time: %.2f minutes' % ((time.time() - time_start) / 60.0))

# Restore the best validation-loss checkpoint before all predictions.
model.load_weights(str(checkpoint_path))

# ============================================================
# 7. Save learning-rate curve and training metrics
# ============================================================
steps_total = int(optimizer.iterations.numpy())
lr_values = [float(lr(step).numpy()) for step in range(steps_total)]
plt.figure(figsize=(7, 4))
plt.plot(range(steps_total), lr_values, color='black', linewidth=2)
plt.xlabel('Optimizer steps')
plt.ylabel('Learning Rate')
plt.tight_layout()
plt.savefig(output_dir / 'Learning_Rate_PS_CALR.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close()

hist = pd.DataFrame(history.history)
hist.insert(0, 'Epoch', np.arange(1, len(hist) + 1))
hist.to_csv(output_dir / 'training_metrics_PS_CALR.csv', index=False)

plt.figure(figsize=(7, 4))
plt.plot(hist['Epoch'], hist['accuracy'], color='blue', linewidth=2,
         label='Training Accuracy')
plt.plot(hist['Epoch'], hist['val_accuracy'], color='red', linewidth=2,
         label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(output_dir / 'Training_Validation_Accuracy.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(hist['Epoch'], hist['loss'], color='blue', linewidth=2,
         label='Training Loss')
plt.plot(hist['Epoch'], hist['val_loss'], color='red', linewidth=2,
         label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(output_dir / 'Training_Validation_Loss.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close()

# ============================================================
# 8. One-dimensional well prediction and evaluation
# ============================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
logits_full = model.predict(X_full, verbose=0)
prob_full = tf.nn.softmax(logits_full, axis=1).numpy()
label_full = np.argmax(prob_full, axis=1).astype(np.int32) + 1
label_true = y_all + 1

test_pred = label_full[test_idx]
test_true = y_test + 1
print('Held-out within-well validation accuracy: %.4f' % accuracy_score(test_true, test_pred))
print('Held-out validation loss: %.6f' % test_loss)
print('Held-out confusion matrix (rows=true, cols=pred, 1/2/3):')
print(confusion_matrix(test_true, test_pred, labels=[1, 2, 3]))
print(classification_report(test_true, test_pred, labels=[1, 2, 3],
                            target_names=['Sandstone', 'mudstone',
                                          'Calcareous sandstone'],
                            zero_division=0))

with open(output_dir / 'Classification_evaluation.txt', 'w', encoding='utf-8') as file:
    file.write('Continuous depth holdout: shallowest 80% train, deepest 20% validation in the SAME well. Not blind-well validation.\n')
    file.write(f'Training depth range: {Depth[train_idx[0]]:.4f} to {Depth[train_idx[-1]]:.4f} m\n')
    file.write(f'Validation depth range: {Depth[test_idx[0]]:.4f} to {Depth[test_idx[-1]]:.4f} m\n')
    file.write('Classes: 1 sandstone, 2 mudstone, 3 calcareous sandstone.\n')
    file.write(f'Held-out validation accuracy: {test_acc:.6f}\n')
    file.write(f'Held-out validation loss: {test_loss:.6f}\n\n')
    file.write('Confusion matrix (rows=true; columns=predicted; 1/2/3):\n')
    file.write(str(confusion_matrix(test_true, test_pred, labels=[1, 2, 3])))
    file.write('\n\n')
    file.write(classification_report(test_true, test_pred, labels=[1, 2, 3],
                                     target_names=['Sandstone', 'mudstone',
                                                   'Calcareous sandstone'],
                                     zero_division=0))

split_marks = np.full(len(Depth), 'Train', dtype=object)
split_marks[test_idx] = 'Validation'
prediction_df = pd.DataFrame({
    'Depth': Depth,
    'True Lith': label_true,
    'Pred Lith': label_full,
    'Data Split': split_marks,
    'Prob Sandstone (1)': prob_full[:, 0],
    'Prob mudstone (2)': prob_full[:, 1],
    'Prob Calcareous sandstone (3)': prob_full[:, 2],
})
prediction_df.to_csv(output_dir / 'Lithology_prediction_1DResNet_PS_CALR.csv', index=False)

# ============================================================
# 9. 1D lithology strips: true vs predicted (not 2D profiles)
# ============================================================
# Use actual depth edges to preserve layer thickness even if sampling varies.
depth_edges = np.empty(len(Depth) + 1, dtype=np.float64)
depth_edges[1:-1] = (Depth[:-1] + Depth[1:]) / 2.0
depth_edges[0] = Depth[0] - (Depth[1] - Depth[0]) / 2.0
depth_edges[-1] = Depth[-1] + (Depth[-1] - Depth[-2]) / 2.0

pad = DEPTH_PAD_RATIO * (Depth.max() - Depth.min())
DEPTH_YLIM = (Depth.max() + pad, Depth.min() - pad)
cmap = ListedColormap(LITH_COLORS)
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)


def draw_lithology(ax, labels, xlabel):
    ax.pcolormesh(
        [LITH_XLIM[0], LITH_XLIM[1]], depth_edges,
        labels[:, np.newaxis], cmap=cmap, norm=norm, shading='flat')
    ax.set_xlim(LITH_XLIM)
    ax.set_ylim(DEPTH_YLIM)
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.tick_params(direction='out', top=False, right=False, width=1.0)


# Side-by-side strip comparison, like the supplied reference image.
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 7), sharey=True)
draw_lithology(ax[0], label_true, 'True lithology')
draw_lithology(ax[1], label_full, 'Predicted lithology')
ax[0].set_ylabel('Depth (m)')
ax[1].tick_params(labelleft=True)
f.subplots_adjust(left=0.18, right=0.96, bottom=0.12, top=0.97, wspace=0.30)
plt.savefig(output_dir / 'Lithology_True_vs_Predicted_1DResNet.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(f)

# Standalone strips for manuscript figure editing.
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
draw_lithology(ax, label_true, 'True lithology')
ax.set_ylabel('Depth (m)')
plt.savefig(output_dir / 'True_Lithology_1D.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(f)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
draw_lithology(ax, label_full, 'Predicted lithology')
ax.set_ylabel('Depth (m)')
plt.savefig(output_dir / 'Predicted_Lithology_1D.tiff', dpi=600,
            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(f)

print('Saved prediction, figures, metrics and checkpoint to:', output_dir)

result = np.column_stack([
    Depth,
    label_true,
    label_full
])

np.savetxt(
    "Lithology_prediction_results_Rock_physics_guided.txt",
    result,
    fmt=["%.4f", "%d", "%d"],
    delimiter="\t",
    header="Depth\tTrue_Lithology\tPredicted_Lithology",
    comments=""
)

print("Lithology prediction results saved to TXT.")

print('Finished one-dimensional 1DResNet lithology classification.')
