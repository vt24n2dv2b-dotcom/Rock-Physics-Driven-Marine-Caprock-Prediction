# -*- coding: utf-8 -*-
"""
caprocks_prediction_GRU_PI_Por_Data_Driven_PS_CALR.py

Data-driven GRU prediction of Poisson impedance (PI) and porosity (por).
Input: EI1 (8 degrees), EI2 (16 degrees).
Output: PI and porosity.
This is the non-physics baseline for the PI/por rpp_two-constrained GRU.

"""

import numpy as np
import scipy.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import time
import pandas as pd
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
plt.rcParams.update({
    "font.family": "Times New Roman",  
    "font.size": 14,           
    "axes.labelsize": 14,     
    "axes.titlesize": 14,    
    "xtick.labelsize": 14,    
    "ytick.labelsize": 14,     
    "legend.fontsize": 14,   
    "figure.titlesize": 14     
})

# Load .mat 文件
Modeltest = io.loadmat('Modeltest.mat')
# Print the variable names available inside the MATLAB file.
print("Keys in the .mat file:", Modeltest.keys())
if 'Modeltest' not in Modeltest:
    raise KeyError("MAT variable 'Modeltest' was not found")
Modeltest = Modeltest['Modeltest'].astype(np.float32)
print("well_py1066 shape:", Modeltest.shape)

if Modeltest.ndim != 2 or Modeltest.shape[1] < 13:
    raise ValueError("The MAT array must be 2D and contain at least 13 columns")

# Inputs: EI1 (8 degrees, MATLAB column 9), EI2 (16 degrees, column 10).
x_train = Modeltest[:, [8, 9]]
# Targets: Poisson impedance (column 12), porosity (column 13).
y_train = Modeltest[:, [11, 12]]
depth1 = Modeltest[:, 0]
print("Input EI1/EI2 shape:", x_train.shape)
print("Output PI/por shape:", y_train.shape)
print("Depth shape:", depth1.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
scaled_x_train = scaler.transform(x_train)
# Normalize the PI and por target columns separately
scaler_label = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
scaled_y_train = scaler_label.transform(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(2, 1),kernel_regularizer='l2'),
    tf.keras.layers.GRU(128, dropout=0.2),
    tf.keras.layers.Dense(2, activation='sigmoid') # Two normalized outputs: PI and porosity
])

model.summary()

# ======================================================
# PS-CALR: warm-up + cosine annealing
class PS_CALR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, max_lr, min_lr, period, warmup):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.period = period
        self.warmup = warmup

    def __call__(self, step):

        step = tf.cast(step, tf.float32)

        # Linear warm-up
        warmup_lr = (
            self.min_lr
            + (self.max_lr - self.min_lr)
            * step
            / self.warmup
        )

        # Cosine annealing after warm-up
        cycle_step = tf.maximum(
            step - self.warmup,
            0
        )

        cosine_lr = (
            self.min_lr
            + 0.5
            * (self.max_lr - self.min_lr)
            * (
                1
                + tf.cos(
                    np.pi
                    * cycle_step
                    / self.period
                )
            )
        )

        return tf.where(
            step < self.warmup,
            warmup_lr,
            cosine_lr
        )

    def get_config(self):

        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "period": self.period,
            "warmup": self.warmup
        }


# Training epochs
epochs = 1000

# PS-CALR parameters
# These parameters are kept consistent with the Physics-GRU implementation.
max_lr = 0.01
min_lr = 1e-5
period = 3000
warmup = 100

lr = PS_CALR(
    max_lr=max_lr,
    min_lr=min_lr,
    period=period,
    warmup=warmup
)

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=lr,
    clipnorm=1.0
)

model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mae', 'mse']
)


class CustomCallback(Callback):

    def on_batch_end(self, batch, logs=None):

        current_step = self.model.optimizer.iterations

        current_lr = self.model.optimizer.learning_rate

        if isinstance(
            current_lr,
            tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            current_lr = current_lr(current_step)

        try:
            current_lr = float(
                tf.keras.backend.get_value(current_lr)
            )
        except Exception:
            current_lr = float(
                lr(current_step).numpy()
            )

        print(
            f"End of batch {batch}, "
            f"learning rate: {current_lr:.10f}"
        )


earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=500,
    verbose=1,
    mode='min'
)




time_start = time.time()
history = model.fit(
    scaled_x_train, scaled_y_train,
    epochs=epochs, verbose=1,
    callbacks=[earlystopper,
               CustomCallback()]
)

time_end = time.time()
print('time cost:', (time_end - time_start) / 60, 'min')

# Generate the PS-CALR learning-rate curve using optimizer-step indices.
# This is only for visualization and does not affect training.
total_optimizer_steps = int(model.optimizer.iterations.numpy())
lr_values = [
    float(lr(step).numpy())
    for step in range(total_optimizer_steps)
]
steps = range(total_optimizer_steps)

plt.plot(steps, lr_values, color='black', linestyle='-', linewidth=2)
plt.xlabel("Optimizer Step",fontsize=14)
plt.ylabel("Learning Rate",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['font.family'] = 'Times New Roman'
plt.show()
# -----------------------------------------------------------------------------

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


print("Start...")
y_train_pred0 = model.predict(scaled_x_train)
print("Prediction completed, starting inverse normalization...")
y_train_pred = scaler_label.inverse_transform(y_train_pred0)
print("Inverse normalization completed, displaying results...")
print("Shape of prediction results:", y_train_pred.shape)
print("First five rows of prediction results:", y_train_pred[:5])
print("Shape of predictions before normalization:", y_train_pred0.shape)

# ------------------Property comparison curves start---------------------
# The plot styling follows the corresponding PI/por Physics-GRU program.
true = y_train
pred = y_train_pred.copy()
pred[:, 0] = np.maximum(pred[:, 0], 0.0)
pred[:, 1] = np.clip(pred[:, 1], 0.0, 1.0)

for i, n in enumerate(["PI", "Porosity"]):
    print(n,
          "MAE", mean_absolute_error(true[:, i], pred[:, i]),
          "RMSE", np.sqrt(mean_squared_error(true[:, i], pred[:, i])),
          "R2", r2_score(true[:, i], pred[:, i]))

fig, ax = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
x_lims = [
    [0 , 1000],
    [0, 0.2]
]

for i, n in enumerate(["PI", "Porosity"]):
    ax[i].plot(true[:, i], depth1, 'k', label="True")
    ax[i].plot(pred[:, i], depth1, 'r--', label="Predicted")
    ax[i].set_xlabel(n)
    ax[i].set_xlim(x_lims[i])

ax[0].set_ylim(np.max(depth1), np.min(depth1))
ax[0].legend()
ax[0].set_ylabel("Depth")
plt.show()
# ------------------Property comparison curves end---------------------

# ------------------Save true/predicted well-log data---------------------
df = pd.DataFrame({
    'Depth': depth1,
    'True PI': true[:, 0],
    'Pred PI': pred[:, 0],
    'True Por': true[:, 1],
    'Pred Por': pred[:, 1]
})
df.to_csv('prediction_data_driven.csv', index=False)
print("Prediction data saved to CSV")

