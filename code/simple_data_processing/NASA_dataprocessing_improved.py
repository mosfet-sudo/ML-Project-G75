import os
import pandas as pd
import numpy as np

# ========== sklearn 只在缩放阶段用，但这里不直接 import StandardScaler ==========
# 使用自定义缩放逻辑：填补 + 标准化 + ε 平滑

# 修正列名为 s1 到 s21（共 21 个传感器）
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = [f's{i}' for i in range(1, 22)]
ALL_FEATS = ['op1', 'op2', 'op3'] + SENSOR_COLS


def get_data_dir():
    """
    自动定位 data/NASA 目录（相对于项目根目录）
    Dynamically determine the absolute path to data/NASA folder
    """
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
    data_dir = os.path.join(project_root, 'data', 'NASA')
    return data_dir

def read_fd(fd: str):
    """
    读取 train 和 test 数据 / Read train_fd and test_fd files
    """
    data_dir = get_data_dir()
    train_path = os.path.join(data_dir, f"train_{fd}.txt")
    test_path  = os.path.join(data_dir, f"test_{fd}.txt")
    train = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLS)
    test  = pd.read_csv(test_path,  sep=r'\s+', header=None, names=COLS)
    # 有些列可能全是空值，dropna 会删除这些列。这里可以保守处理。
    train = train.dropna(axis=1, how='all')
    test  = test.dropna(axis=1, how='all')
    return train, test

def read_rul(fd: str):
    """
    读取 RUL 文件 / Read official RUL file
    """
    data_dir = get_data_dir()
    rul_path = os.path.join(data_dir, f"RUL_{fd}.txt")
    rul = pd.read_csv(rul_path, header=None, names=['RUL'])
    return rul

def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    给训练集每行添加 RUL 列 / Add RUL to training set
    """
    max_cyc = df.groupby('unit')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cyc, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

def build_test_labels(test_df: pd.DataFrame, rul_vec: pd.DataFrame) -> pd.DataFrame:
    """
    用官方 RUL 和测试集最后时刻构造测试标签 / Build test labels per unit
    返回 DataFrame 包含 ['unit','RUL']
    """
    last = test_df.groupby('unit')['cycle'].max().rename('last_cycle').reset_index()
    last = last.sort_values('unit').reset_index(drop=True)
    last['RUL'] = rul_vec['RUL'].values
    return last[['unit', 'RUL']]

def scale_by_train(train_df: pd.DataFrame, test_df: pd.DataFrame, eps: float = 1e-8):
    """
    填补缺失 + 标准化 + ε 平滑 / Fill missing, scale features, add epsilon to avoid divide by zero
    返回 train_scaled, test_scaled, scaler_params = (means, stds_adj)
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # —— 1. 均值填补（用训练集均值填补 train & test 的缺失值）
    feat_means = train_df[ALL_FEATS].mean()
    train_scaled[ALL_FEATS] = train_scaled[ALL_FEATS].fillna(feat_means)
    test_scaled[ALL_FEATS]  = test_scaled[ALL_FEATS].fillna(feat_means)

    # —— 2. 计算标准差 & 平滑
    stds = train_scaled[ALL_FEATS].std(ddof=0)
    stds_adj = stds.copy()
    stds_adj[stds_adj < eps] = eps

    # —— 3. 标准化 (x - mean) / stds_adj
    for feat in ALL_FEATS:
        train_scaled[feat] = (train_scaled[feat] - feat_means[feat]) / stds_adj[feat]
        test_scaled[feat]  = (test_scaled[feat]  - feat_means[feat]) / stds_adj[feat]

    scaler_params = (feat_means, stds_adj)
    return train_scaled, test_scaled, scaler_params

def process_fd(fd: str):
    """
    为一个 FD 子集做完整处理（读取、加 RUL、缩放、保存） / Process one FD subset end-to-end
    """
    train, test = read_fd(fd)
    rul_test = read_rul(fd)

    train = add_train_rul(train)
    test_labels = build_test_labels(test, rul_test)

    # 特征工程可在这里插入

    train_scaled, test_scaled, scaler_params = scale_by_train(train, test)

    # 提取 X, y
    X_train = train_scaled[['unit', 'cycle'] + ALL_FEATS].copy()
    y_train = train_scaled['RUL'].copy()
    X_test  = test_scaled[['unit', 'cycle'] + ALL_FEATS].copy()
    y_test  = test_labels.set_index('unit')['RUL']

    # 保存路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    output_dir = os.path.join(project_root, 'data', 'processed', fd)
    os.makedirs(output_dir, exist_ok=True)

    # 保存 CSV
    X_train.to_csv(os.path.join(output_dir, f"X_train_{fd}.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"y_train_{fd}.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"X_test_{fd}.csv"), index=False)
    test_labels.to_csv(os.path.join(output_dir, f"y_test_{fd}_units.csv"), index=False)

    print(f"[{fd}] processed. Saved to {output_dir}. Scaler params (means, stds_adj):")
    # 可以只打印部分 params，以避免输出过多内容
    means, stds_adj = scaler_params
    print(" means (first 5):", means.head(5).to_dict())
    print(" stds_adj (first 5):", stds_adj.head(5).to_dict())

def main():
    """
    主流程入口 / Main entry: 可以处理多个 FD 子集
    """
    fds = ["FD001", "FD002", "FD003", "FD004"]
    for fd in fds:
        print("Processing", fd)
        try:
            process_fd(fd)
        except Exception as e:
            print(f"Error processing {fd}:", e)

if __name__ == "__main__":
    main()
