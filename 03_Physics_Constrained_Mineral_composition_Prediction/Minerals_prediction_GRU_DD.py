# -*- coding: utf-8 -*-
"""
Created on 22/11/2024

@author: zx
"""

import numpy as np
import scipy.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import time
import pandas as pd
import pandas as pd
from tensorflow.keras.callbacks import Callback
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

Modeltest = io.loadmat('Modeltest.mat')

print("Keys in the .mat file:", Modeltest.keys())

Modeltest = Modeltest.get('Modeltest').astype(np.float32)
print("Modeltest shape:", Modeltest.shape)

x_train = Modeltest[:, 8:13]
y_train = Modeltest[:, 2:5]
depth1 = Modeltest[:, 0]
print(x_train.shape, y_train.shape, depth1.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
scaled_x_train = scaler.transform(x_train)

scaler_label = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
scaled_y_train = scaler_label.transform(y_train)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5, 1),kernel_regularizer='l2'),
    tf.keras.layers.GRU(128, dropout=0.2),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

model.summary()

# ======================================================
# PS-CALR: warm-up + cosine annealing
# The learning-rate schedule is driven directly by optimizer.iterations.
# Therefore, the learning rate is updated at every real optimizer step,
# without manually converting epoch numbers into training steps.
# ======================================================

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
max_lr = 0.003
min_lr = 1e-5
period = 2000
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


total_optimizer_steps = int(model.optimizer.iterations.numpy())
lr_values = [
    float(lr(step).numpy())
    for step in range(total_optimizer_steps)
]
steps = range(total_optimizer_steps)


plt.plot(steps, lr_values, color='black', linestyle='-', linewidth=2)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Learning Rate",fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['font.family'] = 'Times New Roman'
plt.show()
# -----------------------------------------------------------------------------

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


print("Starting prediction...")
y_train_pred0 = model.predict(scaled_x_train)
print("Prediction completed, starting inverse normalization...")
y_train_pred = scaler_label.inverse_transform(y_train_pred0)
print("Inverse normalization completed, displaying results...")
print("Shape of prediction results:", y_train_pred.shape)
print("First five rows of prediction results:", y_train_pred[:5])
print("Shape of predictions before inverse normalization:", y_train_pred0.shape)


f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
ax[0].plot(y_train[:, 0], depth1, color='black', label='True Vquar')
ax[0].plot(y_train_pred[:, 0], depth1, '--', color='r', label='Pred Vquar')
ax[0].invert_yaxis()
ax[0].set_xlim(0.5, 1)
ax[0].set_xlabel('Vquar(%)')
ax[0].legend(loc='upper center')

ax[1].plot(y_train[:, 2], depth1, color='black', label='True Vclc')
ax[1].plot(y_train_pred[:, 2], depth1, '--', color='r', label='Pred Vclc')
ax[1].invert_yaxis()
ax[1].set_xlim(0, 0.5)
ax[1].set_xlabel('Vclc(%)')
ax[1].legend(loc='upper center')

ax[2].plot(y_train[:, 1], depth1, color='black', label='True Vclay')
ax[2].plot(y_train_pred[:, 1], depth1, '--', color='r', label='Pred Vclay')
ax[2].invert_yaxis()
ax[2].set_xlim(0, 0.35)
ax[2].set_xlabel('Vclay(%)')
ax[2].legend(loc='upper center')
plt.legend(loc='upper center')
plt.show()

df = pd.DataFrame({
    'Depth': depth1,

    'True Vquar': y_train[:, 0],
    'Pred Vquar': y_train_pred[:, 0],

    'True Vclc': y_train[:, 2],
    'Pred Vclc': y_train_pred[:, 2],

    'True Vclay': y_train[:, 1],
    'Pred Vclay': y_train_pred[:, 1],

})


df.to_csv('prediction_data_driven.csv', index=False)

print("Prediction data has been saved as a CSV file.")





