# -*- coding: utf-8 -*-
"""
caprocks_prediction_GRU_Physics_Constraint_v8_onestage.py

One-stage GRU that outputs ALL THREE mineral volume fractions at once
(Vquartz, Vcalcite, Vclay). All three enter the SAME differentiable rock
physics forward, so all three are constrained by the elastic EI data.

The rock physics forward is a TensorFlow port of the PI-BDD-GAN model
(rock_physics_torch.py): a Xu-White style DEM + Gassmann chain.
  * fsoft log  -> phi_soft = por*fsoft ; phi_hard = por*(1-fsoft)
  * alpha log  -> Berryman P,Q geometric factors (compute_pq)
  * hard pores -> Khard = Km*(1-phi_hard)^P , Ghard = Gm*(1-phi_hard)^Q
  * pressure  -> kappa = 0.027*ln(Pe_MPa) + 0.065
                 Kdry  = Khard/(1 + phi_soft/kappa)
                 Gdry  = Ghard*Kdry/Khard
  * Wood fluid-> Kf = 1/(sw/Kw + (1-sw)/Kg)
  * Gassmann  -> Ksat (reference = Khard, total porosity por), Gsat = Gdry
  * Vp, Vs    -> Connolly EI at 8 / 16 / 24 deg

"""

import numpy as np
import tensorflow as tf
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



PRESSURE_IS_MPA = True
ALPHA_MIN = 0.001
ALPHA_MAX = 0.99


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

Pe = data[:, 1]

Vq = data[:, 2]
Vclay = data[:, 3]
Vc = data[:, 4]

EI1 = data[:, 8]
EI2 = data[:, 9]
EI3 = data[:, 10]

PI = data[:, 11]
por = data[:, 12]

sw = data[:, 13]

rho = data[:, 14]
vp = data[:, 15]
vs = data[:, 16]

fsoft = data[:, 17]
alpha = data[:, 18]

# Effective pressure in Pa
Pe_pa = Pe * 1e6 if PRESSURE_IS_MPA else Pe


X = np.column_stack([
    EI1,
    EI2,
    EI3,
    PI,
    por
])


Y = np.column_stack([
    Vq,
    Vc,
    Vclay
])


sx = MinMaxScaler()
sy = MinMaxScaler()

Xn = sx.fit_transform(X).astype(np.float32)
Yn = sy.fit_transform(Y).astype(np.float32)

Xn = Xn.reshape((-1, 5, 1))


# ======================================================
# One-stage GRU: three mineral outputs
# ======================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(5, 1),kernel_regularizer='l2'),
    tf.keras.layers.GRU(128, dropout=0.2),
    tf.keras.layers.Dense(3, activation="sigmoid")
])


lr = PS_CALR(
    max_lr=0.003,
    min_lr=1e-5,
    period=2000,
    warmup=100
)

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=lr,
    clipnorm=1.0
)

# ======================================================
# Rock physics constants
# ======================================================

# Mineral moduli (Pa): quartz, calcite, clay
Kq, Gq = 36.6e9, 45e9
Kc, Gc = 76.8e9, 32e9
Kcl, Gcl = 21e9, 7e9

# Fluid moduli: PI-BDD-GAN MODULI_GPA[6]=water [2.0,0], [7]=gas [0.7,0]
Kw = 2.0e9
Kg = 0.7e9


# ======================================================
# Rock physics

def safe_den(x, eps=1e-12):
    sign = tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))
    return tf.where(tf.abs(x) < eps, sign * eps, x)


def compute_pq(km, um, alpha):
    """Berryman P,Q DEM geometric factors from the pore aspect ratio."""
    # Empty-pore end member: Ki=0, mui=0 -> A=-1, B=0
    A = -1.0
    B = 0.0

    al = tf.clip_by_value(alpha, 1e-6, 1.0 - 1e-6)

    R = 3.0 * um / safe_den(3.0 * km + 4.0 * um)

    one_minus = tf.maximum(1.0 - al * al, 1e-12)

    theta = al * (tf.math.acos(al) - al * tf.sqrt(one_minus)) / tf.pow(one_minus, 1.5)
    f = al * al * (3.0 * theta - 2.0) / one_minus

    F1 = 1.0 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = (1.0 + A * (1.0 + 1.5 * (f + theta) - 0.5 * R * (3.0 * f + 5.0 * theta))
          + B * (3.0 - 4.0 * R)
          + 0.5 * A * (A + 3.0 * B) * (3.0 - 4.0 * R)
          * (f + theta - R * (f - theta + 2.0 * theta * theta)))
    F3 = 1.0 + 0.5 * A * (R * (2.0 - theta) + (1.0 + al * al) * f * (R - 1.0) / (al * al))
    F4 = 1.0 + 0.25 * A * (3.0 * theta + f - R * (f - theta))
    F5 = A * (R * (f + theta - 4.0 / 3.0) - f) + B * theta * (3.0 - 4.0 * R)
    F6 = 1.0 + A * (1.0 + f - R * (theta + f)) + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F7 = 2.0 + 0.25 * A * (9.0 * theta + 3.0 * f - R * (5.0 * theta + 3.0 * f)) \
        + B * theta * (3.0 - 4.0 * R)
    F8 = A * (1.0 - 2.0 * R + 0.5 * f * (R - 1.0) + 0.5 * theta * (5.0 * R - 3.0)) \
        + B * (1.0 - theta) * (3.0 - 4.0 * R)
    F9 = A * (f * (R - 1.0) - R * theta) + B * theta * (3.0 - 4.0 * R)

    Tvol = 3.0 * F1 / safe_den(F2)
    Tdev = (2.0 / safe_den(F3)
            + 1.0 / safe_den(F4)
            + (F4 * F5 + F6 * F7 - F8 * F9) / safe_den(F2 * F4))

    P = tf.clip_by_value(Tvol / 3.0, 1e-4, 300.0)
    Q = tf.clip_by_value(Tdev / 5.0, 1e-4, 300.0)

    return P, Q


def vrh_three_minerals(m):
    """Voigt-Reuss-Hill matrix moduli for [Vq, Vc, Vclay]; minerals normalized."""
    c = tf.maximum(m, 0.0)
    c = c / tf.maximum(tf.reduce_sum(c, axis=1, keepdims=True), 1e-12)

    Kj = tf.stack([Kq, Kc, Kcl])
    Gj = tf.stack([Gq, Gc, Gcl])

    Kv = tf.reduce_sum(c * Kj[None, :], axis=1)
    Gv = tf.reduce_sum(c * Gj[None, :], axis=1)

    Kr = 1.0 / tf.reduce_sum(c / Kj[None, :], axis=1)
    Gr = 1.0 / tf.reduce_sum(c / Gj[None, :], axis=1)

    return 0.5 * (Kv + Kr), 0.5 * (Gv + Gr)


def kappa_from_pressure(pe_pa):

    pe_mpa = pe_pa / 1e6
    # kappa = 0.0047 * pe_mpa + 0.001
    kappa = 0.065 + 0.027 * np.log(pe_mpa)
    return tf.maximum(kappa, 1e-12)


def rock_physics_forward(m):
    """m = [Vq,Vc,Vclay] physical fractions -> EI at 3 angles (differentiable)."""

    # Matrix (VRH)
    Km, Gm = vrh_three_minerals(m)

    por_tf = tf.constant(por, dtype=tf.float32)
    fsoft_tf = tf.constant(fsoft, dtype=tf.float32)
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    alpha_tf = tf.clip_by_value(alpha_tf, ALPHA_MIN, ALPHA_MAX)

    # P,Q from the aspect-ratio log
    P, Q = compute_pq(Km, Gm, alpha_tf)

    # Pore partition from fsoft
    phi_soft = por_tf * fsoft_tf
    phi_hard = por_tf * (1.0 - fsoft_tf)

    # Hard pores: DEM power law
    base = tf.maximum(1.0 - phi_hard, 1e-8)
    Khard = Km * tf.pow(base, P)
    Ghard = Gm * tf.pow(base, Q)

    # Soft pores with pressure kappa
    pe_tf = tf.constant(Pe_pa, dtype=tf.float32)
    kappa = kappa_from_pressure(pe_tf)

    Kdry = Khard / (1.0 + phi_soft / kappa)
    Gdry = Ghard * Kdry / tf.maximum(Khard, 1e-12)

    # Wood fluid mixing (pf=-1)
    sw_tf = tf.constant(sw, dtype=tf.float32)
    Kf = 1.0 / (sw_tf / Kw + (1.0 - sw_tf) / Kg)

    # Gassmann with Khard as the solid reference, total porosity por
    denom = (por_tf / tf.maximum(Kf, 1e-12)
             + (1.0 - por_tf) / tf.maximum(Khard, 1e-12)
             - Kdry / tf.maximum(Khard * Khard, 1e-24))
    denom = safe_den(denom)

    Ksat_formula = Kdry + (1.0 - tf.square(Kdry / tf.maximum(Khard, 1e-12))) / denom
    Ksat = tf.where(phi_soft <= 1e-12, Kdry, Ksat_formula)
    Gsat = Gdry

    Ksat = tf.maximum(Ksat, 1e4)
    Gsat = tf.maximum(Gsat, 1e4)

    # Velocities
    rho_tf = tf.constant(rho, dtype=tf.float32)

    Vp = tf.sqrt(tf.maximum((Ksat + 4.0 * Gsat / 3.0) / rho_tf, 1e-12))
    Vs = tf.sqrt(tf.maximum(Gsat / rho_tf, 1e-12))

    # Connolly elastic impedance
    vp0 = tf.reduce_mean(Vp)
    vs0 = tf.reduce_mean(Vs)
    rho0 = tf.reduce_mean(rho_tf)

    K = 0.25

    def EI(theta):

        a = 1 + tf.math.tan(theta) ** 2
        b = -8 * K * tf.sin(theta) ** 2
        c = 1 - 4 * K * tf.sin(theta) ** 2

        d = (
            -tf.math.tan(theta) ** 2 * tf.math.log(vp0)
            +
            8 * K * tf.sin(theta) ** 2 * tf.math.log(vs0)
            +
            4 * K * tf.sin(theta) ** 2 * tf.math.log(rho0)
        )

        return (
            Vp ** a *
            Vs ** b *
            rho_tf ** c *
            tf.exp(d)
        )

    return tf.stack([
        EI(8 * np.pi / 180),
        EI(16 * np.pi / 180),
        EI(24 * np.pi / 180)
    ], axis=1)


def reflection_loss(a, b):

    loss = 0

    for i in range(3):

        rt = (a[1:, i] - a[:-1, i]) / (a[1:, i] + a[:-1, i] + 1e-12)
        rp = (b[1:, i] - b[:-1, i]) / (b[1:, i] + b[:-1, i] + 1e-12)

        loss += tf.reduce_mean(tf.square(rt - rp))

    return loss / 3


def calcite_edge_loss(pred, true):
    gp = pred[1:] - pred[:-1]
    gt = true[1:] - true[:-1]
    w = tf.abs(gt) + 1e-6
    cos = tf.reduce_sum(gp * gt) / (
        tf.sqrt(tf.reduce_sum(gp ** 2)) * tf.sqrt(tf.reduce_sum(gt ** 2)) + 1e-6
    )
    return tf.reduce_mean(w * (1 - cos))


def calcite_peak_loss(pred, true):
    mask = tf.cast(true > 0.1, tf.float32) * 4.0 + 1.0
    return tf.reduce_mean(mask * tf.square(pred - true))


# ======================================================
# Training
# ======================================================

Xtf = tf.constant(Xn)
Ytf = tf.constant(Yn)

EItrue = tf.constant(
    np.column_stack([EI1, EI2, EI3]),
    dtype=tf.float32
)

scale = tf.constant(sy.data_max_ - sy.data_min_, dtype=tf.float32)
offset = tf.constant(sy.data_min_, dtype=tf.float32)


lambda_R = 9
lambda_q = 2
lambda_c = 2
lambda_clay = 2
lambda_sum = 0
lambda_edge = 0
lambda_peak = 0

best = 1e9
patience = 500
wait = 0
history = []


# Diagnostic: confirm logs enter the model and their physical scale.
_phi_soft_diag = por * fsoft
_phi_hard_diag = por * (1.0 - fsoft)
_pe_mpa_diag = Pe_pa / 1e6
print("fsoft   min/mean/max: %.4f %.4f %.4f"
      % (fsoft.min(), fsoft.mean(), fsoft.max()))
print("alpha   min/mean/max: %.5f %.5f %.5f"
      % (alpha.min(), alpha.mean(), alpha.max()))
print("Pe MPa  min/mean/max: %.3f %.3f %.3f"
      % (_pe_mpa_diag.min(), _pe_mpa_diag.mean(), _pe_mpa_diag.max()))
print("phi_soft min/mean/max: %.4f %.4f %.4f"
      % (_phi_soft_diag.min(), _phi_soft_diag.mean(), _phi_soft_diag.max()))
print("phi_hard min/mean/max: %.4f %.4f %.4f"
      % (_phi_hard_diag.min(), _phi_hard_diag.mean(), _phi_hard_diag.max()))


for epoch in range(1000):

    with tf.GradientTape() as tape:

        mm = model(Xtf, training=True)

        q_pred = mm[:, 0]
        c_pred = mm[:, 1]
        clay_pred = mm[:, 2]

        mineral = tf.stack([q_pred, c_pred, clay_pred], axis=1)
        mineral_phys = mineral * (scale + 1e-12) + offset

        EIpred = rock_physics_forward(mineral_phys)

        LR = reflection_loss(EItrue, EIpred)

        Lq = tf.reduce_mean(tf.square(q_pred - Ytf[:, 0]))
        Lc = tf.reduce_mean(tf.square(c_pred - Ytf[:, 1]))
        Lcl = tf.reduce_mean(tf.square(clay_pred - Ytf[:, 2]))

        Ledge = calcite_edge_loss(c_pred, Ytf[:, 1])
        Lpeak = calcite_peak_loss(c_pred, Ytf[:, 1])

        Lsum = tf.reduce_mean(
            tf.square(
                tf.maximum(
                    tf.reduce_sum(mineral_phys, axis=1) - 1, 0)
            )
        )

        loss = (lambda_R * LR +
                lambda_q * Lq +
                lambda_c * Lc +
                lambda_clay * Lcl +
                lambda_sum * Lsum +
                lambda_edge * Ledge +
                lambda_peak * Lpeak)

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    history.append(float(loss))

    if loss < best:
        best = float(loss)
        wait = 0
        model.save("best_GRU_v8_onestage.keras")
    else:
        wait += 1

    if epoch % 10 == 0:
        print(epoch, float(loss), float(LR), float(Lc),
              float(Ledge), float(Lpeak))

    if wait > patience:
        print("Early stopping")
        break


# ======================================================
# Prediction
# ======================================================

mp = model.predict(Xn)

pred = sy.inverse_transform(mp)
pred = np.clip(pred, 0, 1)

true = np.column_stack([Vq, Vc, Vclay])

for i, n in enumerate(["Vquartz", "Vcalcite", "Vclay"]):
    print(n,
          "MAE", mean_absolute_error(true[:, i], pred[:, i]),
          "RMSE", np.sqrt(mean_squared_error(true[:, i], pred[:, i])),
          "R2", r2_score(true[:, i], pred[:, i]))


fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharey=True)
x_lims = [
    [0.5, 1],   # Vquartz
    [0, 0.5],  # Vcalcite
    [0, 0.35]   # Vclay
]
for i, n in enumerate(["Vquartz", "Vcalcite", "Vclay"]):
    ax[i].plot(true[:, i], depth, 'k', label="True")
    ax[i].plot(pred[:, i], depth, 'r--', label="Predicted")
    ax[i].invert_yaxis()
    ax[i].set_xlabel(n)
    ax[i].set_xlim(x_lims[i])
ax[0].legend()
ax[0].set_ylabel("Depth")
plt.show()


import pandas as pd

df = pd.DataFrame({
    'Depth': depth,
    'True Vquar': true[:, 0],
    'Pred Vquar': pred[:, 0],
    'True Vclc': true[:, 1],    
    'Pred Vclc': pred[:, 1],
    'True Vclay': true[:, 2],
    'Pred Vclay': pred[:, 2],
})

# 保存为 CSV 文件
df.to_csv('prediction_physics_constrained.csv', index=False)

print("Prediction data has been saved as a CSV file.")