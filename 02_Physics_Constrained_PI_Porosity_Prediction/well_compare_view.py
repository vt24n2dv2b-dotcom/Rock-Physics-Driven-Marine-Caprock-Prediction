import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
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


df = pd.read_csv('prediction_data_driven.csv')
num_samples = len(df)
y_train_DD = np.zeros((num_samples, 3))
y_train_DD_pred = np.zeros((num_samples, 3))
depth1 = df['Depth'].values

y_train_DD[:, 0] = df['True PI'].values
y_train_DD_pred[:, 0] = df['Pred PI'].values

y_train_DD[:, 1] = 100*df['True Por'].values
y_train_DD_pred[:, 1] = 100*df['Pred Por'].values

df = pd.read_csv('prediction_physics_constrained.csv')
num_samples = len(df)
y_train = np.zeros((num_samples, 3))
y_train_pred = np.zeros((num_samples, 3))
depth1 = df['Depth'].values

y_train[:, 0] = df['True PI'].values
y_train_pred[:, 0] = df['Pred PI'].values

y_train[:, 1] = 100*df['True Por'].values
y_train_pred[:, 1] = 100*df['Pred Por'].values


df = pd.read_csv('PI_por_Bayesian_results.csv')
num_samples = len(df)
y_train_conventional = np.zeros((num_samples, 3))
y_train_conventional_pred = np.zeros((num_samples, 3))
depth1_conventional = df['Depth'].values

y_train_conventional[:, 0] = df['True PI'].values
y_train_conventional_pred[:, 0] = df['Pred PI'].values

y_train_conventional[:, 1] = 100*df['True Por'].values
y_train_conventional_pred[:, 1] = 100*df['Pred Por'].values



print("Depth:", depth1[:5])



#---------a-----------
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(y_train[:, 0], depth1, color='gray', linewidth=1, label='True PI')
ax.plot(y_train_pred[:, 0], depth1, '-', color='r',linewidth=1, label='Pred PI')
ax.plot(y_train_conventional_pred[:, 0], depth1, '-', color='g', alpha=1,linewidth=1,label='Pred PI')
ax.plot(y_train_DD_pred[:, 0], depth1, '-', color='#9900fa', alpha=1, linewidth=1,label='Pred PI')

ax.set_xlim(300,700)
# ax.invert_yaxis()
ax.set_ylim(depth1.max(), depth1.min())
ax.set_ylabel('Depth (m)')
ax.set_xlabel('PI (kg/m³)')



plt.savefig(
        "PI_Por_a.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(y_train[:, 1], depth1, color='gray', linewidth=1, label='True Por')
ax.plot(y_train_pred[:, 1], depth1, '-', color='r',linewidth=1, label='Pred Por')
ax.plot(y_train_conventional_pred[:, 1], depth1, '-', color='g', alpha=1,linewidth=1,label='Pred Por')
ax.plot(y_train_DD_pred[:, 1], depth1, '-', color='#9900fa', alpha=1, linewidth=1,label='Pred Por')

ax.set_xlim(0,15)
# ax.invert_yaxis()
ax.set_ylim(depth1.max(), depth1.min())
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Porosity (%)')



plt.savefig(
        "PI_Por_b.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()




error_GRU_Physics_Constraint = y_train_pred - y_train
error_conventional = y_train_conventional_pred - y_train_conventional
error_GRU_DD = y_train_DD_pred - y_train_DD


param_names = [ 'PI', 'Porosity']
param_units = [ '((kg/m³)(m/s))', '(%)']


colors = ['g', 'r', '#9900fa']  # conventional, GRU, GRU_DD
method_names = ['conventional', 'GRU', 'GRU_DD']


for i in range(2):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))


    ax.plot(error_GRU_DD[:, i], depth1, color=colors[2], linewidth=1,
            label=f'{method_names[2]} Error')

    ax.plot(error_conventional[:, i], depth1, color=colors[0], linewidth=1,
            label=f'{method_names[0]} Error')

    ax.plot(error_GRU_Physics_Constraint[:, i], depth1, color=colors[1], linewidth=1,
            label=f'{method_names[1]} Error')


    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)



    ax.set_ylim(depth1.max(), depth1.min())
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel(f'{param_names[i]} Error {param_units[i]}')



    all_errors = np.concatenate([
        error_conventional[:, i],
        error_GRU_Physics_Constraint[:, i],
        error_GRU_DD[:, i]
    ])
    error_range = np.max(np.abs(all_errors)) * 1.1
    ax.set_xlim(-error_range, error_range)


    plt.savefig(
        f"{param_names[i]}_Error_Comparison.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.show()

# -----------------------------------
from scipy.stats import pearsonr, spearmanr
print("\nComprehensive error statistics of all methods (Conventional vs GRU vs GRU_DD):")
print("=" * 100)

for i, param in enumerate(param_names):
    print(f"\n{param} comprehensive evaluation statistics:")
    print("-" * 85)
    print(
        f"{'Method':<12} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'Pearson':<10} {'Spearman':<10} {'Mean Error':<12}")
    print("-" * 85)

    # Get true values and predictions from different methods
    y_true = y_train[:, i]
    predictions = [
        y_train_conventional_pred[:, i],  # Conventional
        y_train_pred[:, i],  # GRU
        y_train_DD_pred[:, i]  # GRU_DD
    ]

    for j, method in enumerate(method_names):
        y_pred = predictions[j]

        # Calculate evaluation metrics
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)

        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate correlation coefficients
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        # Mean error (with positive/negative sign)
        mean_error = np.mean(y_pred - y_true)

        print(
            f"{method:<12} {mae:>8.4f} {mse:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pearson_corr:>8.4f} {spearman_corr:>8.4f} {mean_error:>10.4f}")

print("\n" + "=" * 100)


# ------------------ Additional: Ranking statistics of the best method ------------------

print("\nRanking of the best methods for each parameter (Conventional vs GRU vs GRU_DD):")
print("=" * 60)

# Find the best method for each parameter
for i, param in enumerate(param_names):
    print(f"\nBest method ranking for {param} (based on RMSE):")
    print("-" * 40)

    y_true = y_train[:, i]
    predictions = [
        y_train_conventional_pred[:, i],  # Conventional
        y_train_pred[:, i],  # GRU
        y_train_DD_pred[:, i]  # GRU_DD
    ]

    method_scores = []
    for j, method in enumerate(method_names):
        y_pred = predictions[j]
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        method_scores.append((method, rmse))

    # Sort by RMSE (ascending)
    method_scores.sort(key=lambda x: x[1])

    for rank, (method, rmse) in enumerate(method_scores, 1):
        print(f"Rank {rank}: {method:<12} RMSE = {rmse:.4f}")

print("\n" + "=" * 60)


# ------------------ Summary table (suitable for paper reporting) ------------------

print("\nSummary table (suitable for paper reporting):")
print("=" * 80)
print(f"{'Parameter':<8} {'Method':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Pearson':<8}")
print("-" * 80)

for i, param in enumerate(param_names):
    y_true = y_train[:, i]
    predictions = [
        y_train_conventional_pred[:, i],  # Conventional
        y_train_pred[:, i],  # GRU
        y_train_DD_pred[:, i]  # GRU_DD
    ]

    for j, method in enumerate(method_names):
        y_pred = predictions[j]

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        pearson_corr, _ = pearsonr(y_true, y_pred)

        print(f"{param:<8} {method:<12} {mae:>6.4f} {rmse:>6.4f} {r2:>6.4f} {pearson_corr:>6.4f}")

print("=" * 80)


# ------------------ Best method statistics for each metric ------------------

print("\nBest method statistics for each metric (Conventional vs GRU vs GRU_DD):")
print("=" * 50)

metrics = ['MAE', 'RMSE', 'R²', 'Pearson']

for metric in metrics:
    print(f"\nBest method for {metric}:")
    print("-" * 30)

    best_methods = {}
    for i, param in enumerate(param_names):
        y_true = y_train[:, i]
        predictions = [
            y_train_conventional_pred[:, i],  # Conventional
            y_train_pred[:, i],  # GRU
            y_train_DD_pred[:, i]  # GRU_DD
        ]

        best_score = float('inf') if metric in ['MAE', 'RMSE'] else float('-inf')
        best_method = ""

        for j, method in enumerate(method_names):
            y_pred = predictions[j]

            if metric == 'MAE':
                score = np.mean(np.abs(y_pred - y_true))
            elif metric == 'RMSE':
                score = np.sqrt(np.mean((y_pred - y_true) ** 2))
            elif metric == 'R²':
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            elif metric == 'Pearson':
                score, _ = pearsonr(y_true, y_pred)

            if (metric in ['MAE', 'RMSE'] and score < best_score) or \
                    (metric in ['R²', 'Pearson'] and score > best_score):
                best_score = score
                best_method = method

        best_methods[param] = (best_method, best_score)

    for param, (method, score) in best_methods.items():
        print(f"{param:<8} {method:<12} {score:>8.4f}")

print("\nError plots and statistical information have been generated successfully!")

