# data_utils.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_and_prepare_data(n_samples=100, slope=2, intercept=5, noise_level=20, test_size=0.2, random_state=42):
    """
    生成、切分並標準化用於線性迴歸的數據。

    Args:
        n_samples (int): 要生成的數據點數量。
        slope (float): 線性關係的斜率 (w)。
        intercept (float): 線性關係的截距 (b)。
        noise_level (float): 添加到數據中的隨機雜訊等級。
        test_size (float): 測試集所佔的比例。
        random_state (int): 用於確保結果可重現的隨機種子。

    Returns:
        tuple: 包含 X_train_scaled, X_test_scaled, y_train, y_test, scaler 物件。
    """
    # 根據線性方程式 y = wX + b 生成數據，並加入雜訊
    np.random.seed(random_state)
    X = 100 * (np.random.rand(n_samples, 1) - 0.5)
    noise = np.random.randn(n_samples, 1) * noise_level
    y = intercept + slope * X + noise

    # 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 特徵縮放 (標準化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler