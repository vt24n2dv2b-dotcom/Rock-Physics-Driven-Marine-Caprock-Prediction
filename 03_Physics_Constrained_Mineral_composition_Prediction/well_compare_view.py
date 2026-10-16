import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    "font.family": "Times New Roman",    # 字体，可以改成 'SimHei'（黑体）以支持中文
    "font.size": 14,           # 全局字体大小
    "axes.labelsize": 14,      # 坐标轴标签（xlabel, ylabel）字体大小
    "axes.titlesize": 14,      # 图标题字体大小
    "xtick.labelsize": 14,     # x 轴刻度字体大小
    "ytick.labelsize": 14,     # y 轴刻度字体大小
    "legend.fontsize": 14,     # 图例字体大小
    "figure.titlesize": 14     # 整个 figure 标题字体大小
})

# 读取 CSV 文件
df = pd.read_csv('prediction_data_driven.csv')
# 初始化 y_train 和 y_train_pred，假设它们有 (样本数, 5) 形状
num_samples = len(df)  # 获取样本数量
y_train_DD = np.zeros((num_samples, 3))  # 初始化全零矩阵
y_train_DD_pred = np.zeros((num_samples, 3))  # 同样初始化预测值矩阵
# 提取每一列的数据
depth1 = df['Depth'].values

y_train_DD[:, 0] = 100*df['True Vquar'].values
y_train_DD_pred[:, 0] = 100*df['Pred Vquar'].values

y_train_DD[:, 1] = 100*df['True Vclc'].values
y_train_DD_pred[:, 1] = 100*df['Pred Vclc'].values

y_train_DD[:, 2] = 100*df['True Vclay'].values
y_train_DD_pred[:, 2] = 100*df['Pred Vclay'].values

df = pd.read_csv('prediction_physics_constrained.csv')
# 初始化 y_train 和 y_train_pred，假设它们有 (样本数, 5) 形状
num_samples = len(df)  # 获取样本数量
y_train = np.zeros((num_samples, 3))  # 初始化全零矩阵
y_train_pred = np.zeros((num_samples, 3))  # 同样初始化预测值矩阵
# 提取每一列的数据
depth1 = df['Depth'].values

y_train[:, 0] = 100*df['True Vquar'].values
y_train_pred[:, 0] = 100*df['Pred Vquar'].values

y_train[:, 1] = 100*df['True Vclc'].values
y_train_pred[:, 1] = 100*df['Pred Vclc'].values

y_train[:, 2] = 100*df['True Vclay'].values
y_train_pred[:, 2] = 100*df['Pred Vclay'].values


df = pd.read_csv('Mineral_Bayesian_results.csv')
# 初始化 y_train 和 y_train_pred，假设它们有 (样本数, 5) 形状
num_samples = len(df)  # 获取样本数量
y_train_conventional = np.zeros((num_samples, 3))  # 初始化全零矩阵
y_train_conventional_pred = np.zeros((num_samples, 3))  # 同样初始化预测值矩阵
# 提取每一列的数据
depth1_conventional = df['Depth'].values

y_train_conventional[:, 0] = 100*df['True Vquar'].values
y_train_conventional_pred[:, 0] = 100*df['Pred Vquar'].values

y_train_conventional[:, 1] = 100*df['True Vclc'].values
y_train_conventional_pred[:, 1] = 100*df['Pred Vclc'].values

y_train_conventional[:, 2] = 100*df['True Vclay'].values
y_train_conventional_pred[:, 2] = 100*df['Pred Vclay'].values


# 打印部分数据检查
print("Depth:", depth1[:5])
# print("True PI:", y_train[:, 0])
# print("Pred PI:", y_train_pred[:, 0])


# ------------------物性对比曲线start---------------------y_train（真实值）和 y_train_pred（预测值）
#---------a-----------
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(y_train[:, 0], depth1, color='gray', linewidth=1, label='True Vquar')
ax.plot(y_train_pred[:, 0], depth1, '-', color='r',linewidth=1, label='Pred Vquar')
ax.plot(y_train_conventional_pred[:, 0], depth1, '-', color='g', alpha=1,linewidth=1,label='Pred Vquar')
ax.plot(y_train_DD_pred[:, 0], depth1, '-', color='#9900fa', alpha=1, linewidth=1,label='Pred Vquar')

ax.set_xlim(50,100)
# ax.invert_yaxis()
ax.set_ylim(depth1.max(), depth1.min())
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vquar (%)')



plt.savefig(
        "Vquar_Vclc_Vclay_duibi_a.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()

#---------b-----------
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(y_train[:, 1], depth1, color='gray', linewidth=1, label='True Vclc')
ax.plot(y_train_pred[:, 1], depth1, '-', color='r',linewidth=1, label='Pred Vclc')
ax.plot(y_train_conventional_pred[:, 1], depth1, '-', color='g', alpha=1,linewidth=1,label='Pred Vclc')
ax.plot(y_train_DD_pred[:, 1], depth1, '-', color='#9900fa', alpha=1, linewidth=1,label='Pred Vclc')

ax.set_xlim(0,50)
# ax.invert_yaxis()
ax.set_ylim(depth1.max(), depth1.min())
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vcalcite (%)')



plt.savefig(
        "Vquar_Vclc_Vclay_duibi_b.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()


# #---------c-----------
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(y_train[:, 2], depth1, color='gray', linewidth=1, label='True Vclay')
ax.plot(y_train_pred[:, 2], depth1, '-', color='r',linewidth=1, label='Pred Vclay')
ax.plot(y_train_conventional_pred[:, 2], depth1, '-', color='g', alpha=1,linewidth=1,label='Pred Vclay')
ax.plot(y_train_DD_pred[:, 2], depth1, '-', color='#9900fa', alpha=1, linewidth=1,label='Pred Vclay')

ax.set_xlim(0,35)
# ax.invert_yaxis()
ax.set_ylim(depth1.max(), depth1.min())
ax.set_xlabel('Vclay (%)')
ax.set_ylabel('Depth (m)')


plt.savefig(
        "Vquar_Vclc_Vclay_duibi_c.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()

# 计算误差
error_GRU_Physics_Constraint = y_train_pred - y_train
error_conventional = y_train_conventional_pred - y_train_conventional
error_GRU_DD = y_train_DD_pred - y_train_DD

# 参数名称和单位
param_names = [ 'Vquar', 'Vcalcite', 'Vclay']
param_units = [ '(%)', '(%)', '(%)']

# 颜色设置（按照绘制顺序）
colors = ['g', 'r', '#9900fa']  # conventional, GRU, GRU_DD
method_names = ['conventional', 'GRU', 'GRU_DD']

# ------------------ 绘制各个参数的误差对比图 ------------------
for i in range(3):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))

    # 按照指定顺序绘制各个方法的误差曲线
    # 1. 画GRU_DD (紫色)
    ax.plot(error_GRU_DD[:, i], depth1, color=colors[2], linewidth=1,
            label=f'{method_names[2]} Error')
    # 2. 画Conventional (绿色)
    ax.plot(error_conventional[:, i], depth1, color=colors[0], linewidth=1,
            label=f'{method_names[0]} Error')
    # 3. 画GRU (红色)
    ax.plot(error_GRU_Physics_Constraint[:, i], depth1, color=colors[1], linewidth=1,
            label=f'{method_names[1]} Error')

    # 添加零误差参考线
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # 设置坐标轴
    # ax.invert_yaxis()
    ax.set_ylim(depth1.max(), depth1.min())
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel(f'{param_names[i]} Error {param_units[i]}')


    # 设置x轴范围（根据误差范围自动调整）
    all_errors = np.concatenate([
        error_conventional[:, i],
        error_GRU_Physics_Constraint[:, i],
        error_GRU_DD[:, i]
    ])
    error_range = np.max(np.abs(all_errors)) * 1.1
    ax.set_xlim(-error_range, error_range)

    # 添加图例
    # ax.legend(loc='upper right', fontsize=8)

    # # 保存图片
    plt.savefig(
        f"{param_names[i]}_Error_Comparison.tiff",
        dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.show()

# ------------------ 打印统计信息 ------------------
from scipy.stats import pearsonr, spearmanr

print("\n各方法综合误差统计信息 (Conventional vs GRU vs GRU_DD):")
print("=" * 100)
for i, param in enumerate(param_names):
    print(f"\n{param} 综合评估统计:")
    print("-" * 85)
    print(
        f"{'Method':<12} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'Pearson':<10} {'Spearman':<10} {'Mean Error':<12}")
    print("-" * 85)

    # 获取真实值和各方法的预测值
    y_true = y_train[:, i]
    predictions = [
        y_train_conventional_pred[:, i],  # Conventional
        y_train_pred[:, i],  # GRU
        y_train_DD_pred[:, i]  # GRU_DD
    ]

    for j, method in enumerate(method_names):
        y_pred = predictions[j]

        # 计算各项指标
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)

        # 计算R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 计算相关系数
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        # 平均误差（带正负号）
        mean_error = np.mean(y_pred - y_true)

        print(
            f"{method:<12} {mae:>8.4f} {mse:>8.4f} {rmse:>8.4f} {r2:>8.4f} {pearson_corr:>8.4f} {spearman_corr:>8.4f} {mean_error:>10.4f}")

print("\n" + "=" * 100)

# ------------------ 额外：最佳方法排名统计 ------------------
print("\n各参数最佳方法排名 (Conventional vs GRU vs GRU_DD):")
print("=" * 60)

# 为每个参数找出最佳方法
for i, param in enumerate(param_names):
    print(f"\n{param} 最佳方法排名 (按RMSE):")
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

    # 按RMSE排序（从小到大）
    method_scores.sort(key=lambda x: x[1])

    for rank, (method, rmse) in enumerate(method_scores, 1):
        print(f"第{rank}名: {method:<12} RMSE = {rmse:.4f}")

print("\n" + "=" * 60)

# ------------------ 汇总表格（适合论文使用） ------------------
print("\n汇总表格（适合论文报告）:")
print("=" * 80)
print(f"{'参数':<8} {'方法':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Pearson':<8}")
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

# ------------------ 各指标的最佳方法统计 ------------------
print("\n各指标最佳方法统计 (Conventional vs GRU vs GRU_DD):")
print("=" * 50)

metrics = ['MAE', 'RMSE', 'R²', 'Pearson']
for metric in metrics:
    print(f"\n{metric} 最佳方法:")
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

print("\n误差图和统计信息已生成完成！")

