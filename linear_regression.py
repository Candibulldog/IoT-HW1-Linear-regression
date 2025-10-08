# linear_regression.py

import numpy as np


class LinearRegression:
    """
    從零開始實作的線性迴歸模型。

    使用梯度下降法進行優化。
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化模型。

        Args:
            learning_rate (float): 學習率，控制每一步更新的大小。
            n_iterations (int): 梯度下降的迭代次數。
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _initialize_parameters(self, n_features):
        """步驟 2.1: 初始化權重和偏置"""
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def _compute_hypothesis(self, X):
        """步驟 2.2: 定義假設函數 h(x) = wX + b"""
        return np.dot(X, self.weights) + self.bias

    def _compute_cost(self, y, y_pred):
        """步驟 2.3: 定義成本函數 (Mean Squared Error)"""
        m = len(y)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost

    def _gradient_descent(self, X, y, y_pred):
        """步驟 2.4: 實作梯度下降"""
        m = len(y)

        # 步驟 2.4.1: 計算梯度
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # 步驟 2.4.2: 更新權重和偏置
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        """
        使用訓練數據來訓練模型。

        Args:
            X (np.ndarray): 訓練數據的特徵，維度為 (n_samples, n_features)。
            y (np.ndarray): 訓練數據的目標值，維度為 (n_samples, 1)。
        """
        # 取得樣本數和特徵數
        n_samples, n_features = X.shape

        # 初始化參數
        self._initialize_parameters(n_features)

        # 開始梯度下降迭代
        for i in range(self.n_iterations):
            # 1. 計算預測值
            y_pred = self._compute_hypothesis(X)

            # 2. 計算成本函數，並記錄下來以供後續分析
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)

            # 3. 執行梯度下降來更新參數
            self._gradient_descent(X, y, y_pred)

            # (可選) 每隔一定次數打印一次成本，方便監控
            # if (i % 100) == 0:
            #     print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X):
        """
        使用訓練好的模型進行預測。

        Args:
            X (np.ndarray): 要預測的數據，維度為 (n_samples, n_features)。

        Returns:
            np.ndarray: 預測結果。
        """
        if self.weights is None or self.bias is None:
            raise Exception("模型尚未訓練，請先調用 fit() 方法。")

        return self._compute_hypothesis(X)


# --- 使用範例 (用於獨立測試此模組) ---
if __name__ == "__main__":
    # 引入我們剛寫好的數據處理工具
    from data_utils import generate_and_prepare_data

    # 1. 準備數據
    X_train, X_test, y_train, y_test, _ = generate_and_prepare_data()

    # 2. 建立並訓練模型
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # 3. 進行預測
    predictions = model.predict(X_test)

    # 4. 打印結果
    print("模型訓練完成。")
    print(f"學習到的權重 (w): {model.weights[0][0]:.4f}")
    print(f"學習到的偏置 (b): {model.bias:.4f}")

    # 簡單比較一下前 5 個預測值和真實值
    print("\n--- 預測 vs. 真實 ---")
    for i in range(5):
        print(f"預測值: {predictions[i][0]:.2f}, 真實值: {y_test[i][0]:.2f}")
