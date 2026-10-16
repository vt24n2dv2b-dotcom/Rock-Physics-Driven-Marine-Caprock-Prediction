# -*- coding: utf-8 -*-
"""
PI_Por_prediction_GRU_RPC.py

One-stage GRU that outputs PI and porosity at once.
Both predictions enter the SAME differentiable rpp_two reflection equation.
The corresponding EI reflection coefficients are estimated from EI1 and EI2.

Input:
    EI1 (8 degrees), EI2 (16 degrees)
Output:
    PI (Poisson impedance), por (porosity)

For well-specific physical inversion, verify/recalibrate fixed F_PI/F_POR values.
"""

import numpy as np
import tensorflow as tf
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ======================================================
# PS-CALR (warmup + cosine annealing)
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

        warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup

        cycle_step = tf.maximum(step - self.warmup, 0)

        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + tf.cos(np.pi * cycle_step / self.period)
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


# ======================================================
# Load data
# ======================================================

mat = io.loadmat("Modeltest.mat")
data = mat["Modeltest"].astype(np.float32)

depth = data[:, 0]

EI1 = data[:, 8]   # 8 degrees, MATLAB column 9
EI2 = data[:, 9]   # 16 degrees, MATLAB column 10

PI = data[:, 11]   # MATLAB column 12
por = data[:, 12]  # MATLAB column 13


X = np.column_stack([
    EI1,
    EI2
])


Y = np.column_stack([
    PI,
    por
])


sx = MinMaxScaler()
sy = MinMaxScaler()

Xn = sx.fit_transform(X).astype(np.float32)
Yn = sy.fit_transform(Y).astype(np.float32)

Xn = Xn.reshape((-1, 2, 1))


# ======================================================
# One-stage GRU: PI and porosity outputs
# ======================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(2, 1),kernel_regularizer='l2'),
    tf.keras.layers.GRU(128, dropout=0.2),
    tf.keras.layers.Dense(2, activation="sigmoid")
])


lr = PS_CALR(
    max_lr=0.01,
    min_lr=1e-5,
    period=3000,
    warmup=100
)

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=lr,
    clipnorm=1.0
)


# ======================================================
# Fixed background-calibration coefficients for 8 and 16 degrees.
# These constants are calculated from the supplied MATLAB example;
# P-wave velocity, S-wave velocity and density are NOT model inputs.
F_PI = tf.constant([1.1927809929533209, 1.0963799815218047], dtype=tf.float32)
F_POR = tf.constant([0.044107046928887145, 0.03969496117466189], dtype=tf.float32)


def rock_physics_forward(m):
    """m = [PI, por] in physical units -> rpp_two at 8 and 16 degrees."""

    PI_pred = m[:, 0]
    por_pred = m[:, 1]

    # Reflectivity is computed between adjacent DEPTH samples, not angles.
    rPI = 2 * (PI_pred[1:] - PI_pred[:-1]) / (
        PI_pred[1:] + PI_pred[:-1] + 1e-12
    )

    rPOR = 2 * (por_pred[1:] - por_pred[:-1]) / (
        por_pred[1:] + por_pred[:-1] + 1e-12
    )

    return tf.stack([
        F_PI[0] * rPI + F_POR[0] * rPOR,
        F_PI[1] * rPI + F_POR[1] * rPOR
    ], axis=1)


def reflection_loss(a, b):
    """a: observed EI at 8/16 degrees; b: theoretical rpp_two."""

    loss = 0

    for i in range(2):

        rt = (a[1:, i] - a[:-1, i]) / (a[1:, i] + a[:-1, i] + 1e-12)
        rp = b[:, i]

        loss += tf.reduce_mean(tf.square(rt - rp))

    return loss / 2


# ======================================================
# Training
# ======================================================

Xtf = tf.constant(Xn)
Ytf = tf.constant(Yn)

EItrue = tf.constant(
    np.column_stack([EI1, EI2]),
    dtype=tf.float32
)

scale = tf.constant(sy.data_max_ - sy.data_min_, dtype=tf.float32)
offset = tf.constant(sy.data_min_, dtype=tf.float32)


lambda_R = 12
lambda_PI = 2
lambda_por = 2

best = 1e9
patience = 500
wait = 0
history = []


for epoch in range(1000):

    with tf.GradientTape() as tape:

        mm = model(Xtf, training=True)

        PI_pred = mm[:, 0]
        por_pred = mm[:, 1]

        PI_por = tf.stack([PI_pred, por_pred], axis=1)
        PI_por_phys = PI_por * (scale + 1e-12) + offset

        Rpp_pred = rock_physics_forward(PI_por_phys)

        LR = reflection_loss(EItrue, Rpp_pred)

        LPI = tf.reduce_mean(tf.square(PI_pred - Ytf[:, 0]))
        Lpor = tf.reduce_mean(tf.square(por_pred - Ytf[:, 1]))

        loss = (lambda_R * LR +
                lambda_PI * LPI +
                lambda_por * Lpor)

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    history.append(float(loss))

    if loss < best:
        best = float(loss)
        wait = 0
        model.save("best_GRU_PI_Por_v8_onestage.keras")
    else:
        wait += 1

    if epoch % 10 == 0:
        print(epoch, float(loss), float(LR), float(LPI), float(Lpor))

    if wait > patience:
        print("Early stopping")
        break


# ======================================================
# Prediction
# ======================================================

mp = model.predict(Xn)

pred = sy.inverse_transform(mp)
pred[:, 0] = np.maximum(pred[:, 0], 0)
pred[:, 1] = np.clip(pred[:, 1], 0, 1)

true = np.column_stack([PI, por])

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
    ax[i].plot(true[:, i], depth, 'k', label="True")
    ax[i].plot(pred[:, i], depth, 'r--', label="Predicted")
    # ax[i].invert_yaxis()
    ax[i].set_xlabel(n)
    ax[i].set_xlim(x_lims[i])
ax[0].set_ylim(np.max(depth), np.min(depth))
ax[0].legend()
ax[0].set_ylabel("Depth")
plt.show()

print("Finished v8 one-stage GRU, PI and porosity physics constrained")
import pandas as pd

# ------------------Save true/prediction comparison data---------------------
# Create the same-style DataFrame as in the original v8 program.
df = pd.DataFrame({
    'Depth': depth,
    'True PI': true[:, 0],
    'Pred PI': pred[:, 0],
    'True Por': true[:, 1],
    'Pred Por': pred[:, 1],
})

# Save to CSV using the original v8 output style.
df.to_csv('prediction_physics_constrained.csv', index=False)
print("Prediction data saved to CSV")

