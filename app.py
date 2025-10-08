# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 引入我們自定義的模組
from data_utils import generate_and_prepare_data
from linear_regression import LinearRegression

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # 確保負號可以正常顯示


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
