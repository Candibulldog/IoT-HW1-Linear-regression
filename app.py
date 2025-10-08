# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_utils import generate_and_prepare_data
from linear_regression import LinearRegression

# Set Matplotlib parameters to prevent font issues
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # Ensure minus sign is displayed correctly


# --- Helper functions for evaluation metrics ---
def mean_squared_error(y_true, y_pred):
    """Calculates Mean Squared Error (MSE)"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """Calculates R-squared (R2 Score)"""
    corr_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())
    corr = corr_matrix[0, 1]
    return corr**2


# --- Streamlit App Interface ---

st.set_page_config(layout="wide")

# 1. Title
st.title("📈 Interactive Linear Regression Visualizer")
st.markdown(
    "This tool lets you explore Linear Regression from scratch. Adjust the parameters on the left sidebar to see how the model changes."
)

# 2. Sidebar for parameter controls
st.sidebar.header("⚙️ Parameters")

st.sidebar.subheader("📊 Data Generation")
n_samples = st.sidebar.slider("Number of samples (N)", 50, 500, 100, 10)
slope = st.sidebar.slider("True slope (w)", -5.0, 5.0, 2.0, 0.1)
intercept = st.sidebar.slider("True intercept (b)", -10.0, 10.0, 5.0, 0.5)
noise_level = st.sidebar.slider("Noise level", 0.0, 50.0, 20.0, 1.0)

st.sidebar.subheader("🧠 Model Training")
learning_rate = st.sidebar.select_slider(
    "Learning Rate", options=[0.0001, 0.001, 0.01, 0.1, 1.0], value=0.01
)
n_iterations = st.sidebar.slider("Number of iterations", 100, 3000, 1000, 100)

# 3. Data preparation and model training
# Generate data based on sidebar parameters
X_train, X_test, y_train, y_test, scaler = generate_and_prepare_data(
    n_samples=n_samples, slope=slope, intercept=intercept, noise_level=noise_level
)

# Create and train the model
model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
model.fit(X_train, y_train)

# Make predictions
y_pred_test = model.predict(X_test)

# 4. Display results
st.header("✨ Results and Analysis")

col1, col2 = st.columns((1, 1))

with col1:
    # Step 3: Monitor convergence
    st.subheader("📉 Cost Function Convergence")

    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(model.n_iterations), model.cost_history)
    ax_cost.set_xlabel("Iterations")
    ax_cost.set_ylabel("Cost (MSE)")
    ax_cost.set_title("Cost Function over Iterations")
    sns.despine(fig=fig_cost)
    st.pyplot(fig_cost)

    st.markdown("""
    The plot above shows how the model's cost (error) decreases as training progresses.
    An ideal learning rate will result in a smooth, downward-sloping curve.
    - If the curve decreases too slowly, try **increasing the learning rate**.
    - If the curve fluctuates wildly or diverges, the learning rate is too high and you should **decrease it**.
    """)

with col2:
    # Step 4: Evaluate and visualize predictions
    st.subheader("🎯 Prediction Visualization")

    fig_pred, ax_pred = plt.subplots()
    # Original data points (test set)
    ax_pred.scatter(
        scaler.inverse_transform(X_test),
        y_test,
        alpha=0.7,
        label="Actual Values",
    )
    # Regression line
    ax_pred.plot(
        scaler.inverse_transform(X_test),
        y_pred_test,
        color="red",
        linewidth=2,
        label="Prediction",
    )
    ax_pred.set_xlabel("Feature X")
    ax_pred.set_ylabel("Target y")
    ax_pred.set_title("Prediction vs. Actual Values")
    ax_pred.legend()
    sns.despine(fig=fig_pred)
    st.pyplot(fig_pred)

    st.markdown("""
    This plot shows the model's performance on the **unseen test data**.
    - The **blue dots** represent the actual data points.
    - The **red line** is the linear relationship learned by our model.
    The closer the red line fits the trend of the blue dots, the better the model's performance.
    """)

# Step 4: Calculate evaluation metrics
st.subheader("📝 Model Evaluation Metrics")
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
metric2.metric(label="R-squared (R² Score)", value=f"{r2:.4f}")

st.info(
    f"Learned weight (w): **{model.weights[0][0]:.4f}** | Learned bias (b): **{model.bias:.4f}**",
    icon="🧠",
)


st.header("📜 Conclusion")
st.markdown("""
This interactive tool demonstrates the core workflow of linear regression. By adjusting the parameters, we can observe the following:
1.  **Data Characteristics**: Increasing the `Noise level` makes the data points more scattered, making it harder for the model to find the best fit and lowering the R² score.
2.  **Model Training**: The `Learning Rate` and `Number of iterations` directly impact the model's convergence. An improper learning rate can prevent the cost from decreasing effectively.
3.  **Implementation from Scratch**: The underlying `LinearRegression` class was built entirely from scratch, successfully implementing the gradient descent optimization process.

We have now completed all the main steps defined in the `Todo.md` file!
""")
