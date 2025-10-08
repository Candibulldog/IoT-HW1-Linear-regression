# AI-Assisted Development Log for Linear Regression Visualizer

本文件記錄了使用 AI 助理協同開發「互動式線性迴歸視覺化工具」的完整過程。開發流程遵循 `Todo.md` 的規劃，並以 `README.md` 的專案目標為指導。

## 階段一：專案啟動與方向確認

此階段的目標是確立最終要交付的成品樣貌，並規劃開發路徑。

### 👤 使用者初始請求

> 請根據這兩份文件 (`Todo.md` 和 `README.md`) 幫我完成所需的程式碼，我們可以一步一步將細節處理好，我們先確認最後要交出的成品，然後再開始作業。

### 🤖 AI 回應與開發方向確認

AI 分析了文件後，總結出專案的核心目標：

1.  建立一個 Streamlit 應用程式 (`app.py`)。
2.  應用程式的核心功能是互動式地展示線性迴歸。
3.  線性迴歸模型必須**從零開始實作**，而非使用現成函式庫。

AI 提出了以下開發藍圖：
> 我們主要專注於撰寫一個 `linear_regression.py` 檔案，其中包含從零開始實作的線性迴歸邏輯，然後再將這個模型整合到一個 `app.py`（Streamlit 應用程式）中來進行視覺化。

### ✅ 方向確認

使用者同意了此開發方向，專案正式進入實作階段。

---

## 階段二：數據準備模組 (`data_utils.py`)

此階段的目標是完成 `Todo.md` 中的「1. Data Preparation」。

### 👤 使用者請求

> 沒問題，我也認為是這個方向，讓我們開始第一步吧。

### 🤖 AI 回應與程式碼實作

根據專案的互動性需求，AI 建議編寫一個能動態生成數據的函數，而非讀取固定檔案。此函數包含了數據生成、切分與特徵縮放的完整流程。

#### 初版程式碼

```python
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
```

### 💡 關鍵架構決策：程式碼模組化

> **👤 使用者提問：** 我確認過了，你覺得要拆分成不同的 py file 會更好嗎?

AI 強烈建議將程式碼拆分到不同檔案中，理由如下：
1.  **關注點分離 (Separation of Concerns)**：`data_utils.py`, `linear_regression.py`, `app.py` 各司其職。
2.  **可讀性與可維護性**：易於尋找、修改和除錯。
3.  **可重用性**：核心模型可被其他專案輕易引用。

最終確立了專案結構：
linear-regression-visualizer/
├── app.py
├── data_utils.py
├── linear_regression.py
└── requirements.txt


#### 最終檔案: `data_utils.py`
```python
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
```
---

## 階段三：核心模型實作 (`linear_regression.py`)

此階段的目標是完成 `Todo.md` 中的「2. Model Implementation」，從零開始打造線性迴歸模型。

### 👤 使用者請求

> 請繼續吧。

### 🤖 AI 回應與程式碼實作

AI 提供了一個 `LinearRegression` 類別，該類別封裝了所有模型相關的邏輯，包括：
-   權重與偏置的初始化
-   假設函數 (Hypothesis)
-   成本函數 (Cost Function - MSE)
-   梯度下降 (Gradient Descent) 的完整優化流程

#### 最終檔案: `linear_regression.py`
```python
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

```
---

## 階段四：Streamlit 介面與整合 (`app.py`)

此階段的目標是完成 `Todo.md` 中剩餘的「Training」、「Evaluation」、「Prediction」和「Reporting」部分，並將所有模組整合到一個互動式 Web 應用中。

### 👤 使用者請求

> 沒問題，請幫我完成這部分。

### 🤖 AI 回應與程式碼實作

AI 提供了 `app.py` 的完整程式碼，其主要功能包括：
1.  **Streamlit 介面**：使用側邊欄讓使用者調整數據和模型參數。
2.  **模組整合**：導入並使用 `data_utils.py` 和 `linear_regression.py`。
3.  **模型訓練與評估**：根據使用者輸入即時訓練模型，並計算 MSE 和 R² 分數。
4.  **結果視覺化**：繪製成本函數收斂曲線圖和預測結果對比圖。
5.  **說明與報告**：在介面上提供 Markdown 格式的圖表說明與結論。

#### 最終檔案: `app.py`
```python
# app.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# 引入我們自定義的模組
from data_utils import generate_and_prepare_data
from linear_regression import LinearRegression


# --- 輔助函數 (用於計算評估指標) ---
def mean_squared_error(y_true, y_pred):
    """計算均方誤差 (MSE)"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """計算 R-squared (R2 Score)"""
    corr_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())
    corr = corr_matrix[0, 1]
    return corr**2


# --- Streamlit 應用程式介面 ---

st.set_page_config(layout="wide")

# 1. 標題
st.title("📈 互動式線性迴歸視覺化工具")
st.markdown(
    "這個工具讓你從零開始探索線性迴歸。你可以在左側的側邊欄調整參數，觀察模型的變化。"
)

# 2. 側邊欄：參數控制
st.sidebar.header("⚙️ 參數設定")

st.sidebar.subheader("📊 數據生成參數")
n_samples = st.sidebar.slider("數據點數量 (N)", 50, 500, 100, 10)
slope = st.sidebar.slider("真實斜率 (w)", -5.0, 5.0, 2.0, 0.1)
intercept = st.sidebar.slider("真實截距 (b)", -10.0, 10.0, 5.0, 0.5)
noise_level = st.sidebar.slider("雜訊等級", 0.0, 50.0, 20.0, 1.0)

st.sidebar.subheader("🧠 模型訓練參數")
learning_rate = st.sidebar.select_slider(
    "學習率 (Learning Rate)", options=[0.0001, 0.001, 0.01, 0.1, 1.0], value=0.01
)
n_iterations = st.sidebar.slider("迭代次數", 100, 3000, 1000, 100)

# 3. 數據準備與模型訓練
# 根據側邊欄的參數生成數據
X_train, X_test, y_train, y_test, scaler = generate_and_prepare_data(
    n_samples=n_samples, slope=slope, intercept=intercept, noise_level=noise_level
)

# 建立並訓練模型
model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
model.fit(X_train, y_train)

# 進行預測
y_pred_test = model.predict(X_test)

# 4. 顯示結果
st.header("✨ 結果與分析")

col1, col2 = st.columns((1, 1))

with col1:
    # 步驟 3: 監控收斂過程
    st.subheader("📉 成本函數收斂過程")

    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(model.n_iterations), model.cost_history)
    ax_cost.set_xlabel("迭代次數 (Iterations)")
    ax_cost.set_ylabel("成本 (Cost - MSE)")
    ax_cost.set_title("Cost Function over Iterations")
    sns.despine(fig=fig_cost)
    st.pyplot(fig_cost)

    st.markdown("""
    上圖顯示了隨著訓練的進行，模型的成本（誤差）如何逐漸降低。
    一個理想的學習率會讓這條曲線平滑地下降並收斂。
    - 如果曲線下降太慢，可以嘗試**提高學習率**。
    - 如果曲線劇烈震盪或發散，表示學習率太高，需要**降低學習率**。
    """)

with col2:
    # 步驟 4: 評估與視覺化預測
    st.subheader("🎯 預測結果視覺化")

    fig_pred, ax_pred = plt.subplots()
    # 原始數據點 (測試集)
    ax_pred.scatter(
        scaler.inverse_transform(X_test),
        y_test,
        alpha=0.7,
        label="真實值 (Actual Values)",
    )
    # 迴歸線
    ax_pred.plot(
        scaler.inverse_transform(X_test),
        y_pred_test,
        color="red",
        linewidth=2,
        label="預測線 (Prediction)",
    )
    ax_pred.set_xlabel("特徵 (Feature X)")
    ax_pred.set_ylabel("目標 (Target y)")
    ax_pred.set_title("Prediction vs. Actual Values")
    ax_pred.legend()
    sns.despine(fig=fig_pred)
    st.pyplot(fig_pred)

    st.markdown("""
    上圖展示了模型在**未見過的測試數據**上的表現。
    - **藍點**是真實的數據分佈。
    - **紅線**是我們的模型學習到的線性關係。
    紅線越能貼近藍點的分佈趨勢，代表模型學得越好。
    """)

# 步驟 4: 計算評估指標
st.subheader("📝 模型評估指標")
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric(label="均方誤差 (MSE)", value=f"{mse:.2f}")
metric2.metric(label="R-squared (R² Score)", value=f"{r2:.4f}")

st.info(
    f"模型學習到的權重 (w): **{model.weights[0][0]:.4f}** | 學習到的偏置 (b): **{model.bias:.4f}**",
    icon="🧠",
)


st.header("📜 結論")
st.markdown("""
這個互動工具展示了線性迴歸的核心流程。透過調整左側的參數，我們可以觀察到：
1.  **數據特性**：增加 `雜訊等級` 會讓數據點更分散，模型更難找到最佳擬合線，導致 R² 分數下降。
2.  **模型訓練**：`學習率` 和 `迭代次數` 直接影響模型的收斂效果。不恰當的學習率會導致成本無法有效降低。
3.  **從零實作**：我們底層使用的 `LinearRegression` 類別是完全從零打造的，成功實現了梯度下降的優化過程。

我們已經成功完成了 `Todo.md` 中定義的所有主要步驟！
""")

```

---
## 結論

透過與 AI 助理的逐步協作，我們成功地按照 `Todo.md` 的規劃，完成了一個功能完整的互動式線性迴歸專案。整個過程涵蓋了從需求確認、架構設計、模組化編碼到最終整合與呈現的完整軟體開發流程。