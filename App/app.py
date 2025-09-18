import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------- Load data -------------
df = pd.read_excel("../WindPowerForecastingData.xlsx")
if 'TIMESTAMP' in df.columns:
    # Adjust format if your timestamps are in a different format
    try:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    except Exception:
        # Fallback: let pandas infer
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')

# ------------- Helper functions -------------
def data_overview():
    """Return the full dataframe and a short text summary."""
    shape_info = f"Data Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    time_info = ""
    if "TIMESTAMP" in df.columns:
        time_info = f"Time Range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}"
    return df, f"{shape_info}\n{time_info}"

def plot_data(x_col, y_col, plot_type):
    """Generic plot for any two columns from df."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_type == "Line":
        ax.plot(df[x_col], df[y_col])
    elif plot_type == "Scatter":
        ax.scatter(df[x_col], df[y_col])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True)
    return fig

def show_stats_all(column):
    """Return stats table + a figure with histogram & boxplot of a numeric column."""
    stats = df[column].describe().to_frame().reset_index()
    stats.columns = ["Statistic", "Value"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df[column].hist(ax=ax[0], bins=30, color='skyblue')
    ax[0].set_title(f"Histogram of {column}")
    df.boxplot(column=column, ax=ax[1])
    ax[1].set_title(f"Boxplot of {column}")
    plt.tight_layout()
    return stats, fig

def get_forecast_file(model_name):
    """Map model selector to file path."""
    if model_name == "BiLSTM":
        return "../Predictions/forecast_results_bilstm.csv"
    elif model_name == "BiGLSTM":
        return "../Predictions/forecast_results_biglstm.csv"
    else:
        return None

def compute_error_metrics_table(df_fore):
    """Compute MAE, MSE, RMSE, R^2 for the full forecast file."""
    y_true = df_fore["Actual"].values
    y_pred = df_fore["Predicted"].values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
        "Value": [mae, mse, rmse, r2]
    })

def plot_forecast(filename, plot_len=2500):
    """
    Plot only the first `plot_len` points,
    but compute/annotate accuracy using the full series.
    """
    df_full = pd.read_csv(filename)
    # Prepare truncated view for plotting
    plot_df = df_full.iloc[:min(plot_len, len(df_full))].reset_index(drop=True)

    # Compute metrics on full data
    metrics_df = compute_error_metrics_table(df_full)
    acc = metrics_df.loc[metrics_df["Metric"] == "R² Score", "Value"].values[0]

    # Determine model name for title
    if filename.endswith("bilstm.csv"):
        name = "BiLSTM"
    elif filename.endswith("biglstm.csv"):
        name = "BiGLSTM"
    else:
        name = "Unknown Model"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["Actual"], label="Actual", linewidth=1.5)
    ax.plot(plot_df["Predicted"], label="Predicted", linewidth=1.5)
    ax.set_title(f"Forecast vs Actual - {name} (Accuracy: {acc*100:.2f}%)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_forecast_model(model_name):
    """
    Button handler for Forecast Analysis.
    Returns metrics table over ALL points + plot over first 2500 points.
    """
    filename = get_forecast_file(model_name)
    if filename is None:
        return pd.DataFrame({"Error": ["Unknown model selected."]}), None

    df_full = pd.read_csv(filename)
    metrics_df = compute_error_metrics_table(df_full)   # metrics on full series
    fig = plot_forecast(filename, plot_len=2500)        # plot on first 2500
    return metrics_df, fig

# ------------- Gradio UI -------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌬 Wind Power Forecasting Data Analysis")

    # --- Data Overview Tab ---
    with gr.Tab("Data Overview"):
        btn_overview = gr.Button("Show Data Overview")
        data_table = gr.Dataframe(label="DataFrame")
        info = gr.Textbox(label="Info", lines=3)
        btn_overview.click(data_overview, outputs=[data_table, info])

    # --- Visualization Tab ---
    with gr.Tab("Visualization"):
        with gr.Row():
            with gr.Column(scale=1):
                x_col = gr.Dropdown(choices=list(df.columns), value=(df.columns[0] if len(df.columns) > 0 else None), label="X-axis")
                default_y = df.columns[1] if len(df.columns) > 1 else (df.columns[0] if len(df.columns) > 0 else None)
                y_col = gr.Dropdown(choices=list(df.columns), value=default_y, label="Y-axis")
                plot_type = gr.Radio(choices=["Line", "Scatter"], value="Line", label="Plot Type")
                plot_btn_vis = gr.Button("Plot")
            with gr.Column(scale=2):
                plot_output_vis = gr.Plot(label="Plot")
        plot_btn_vis.click(plot_data, inputs=[x_col, y_col, plot_type], outputs=plot_output_vis)

    # --- Statistics Tab ---
    with gr.Tab("Statistics"):
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        default_num = numeric_cols[0] if len(numeric_cols) > 0 else None
        col_choice = gr.Dropdown(choices=numeric_cols, value=default_num, label="Select Column")
        stats_btn = gr.Button("Show Stats")
        stats_output = gr.Dataframe(label="Summary Statistics")
        stats_plot = gr.Plot(label="Histogram & Boxplot")
        stats_btn.click(show_stats_all, inputs=col_choice, outputs=[stats_output, stats_plot])

    # --- Forecast Analysis Tab ---
    with gr.Tab("Forecast Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                forecast_file = gr.Dropdown(["BiLSTM", "BiGLSTM"], label="Choose Model")
                plot_btn_forecast = gr.Button("Show Forecast Results")
                metrics_table = gr.Dataframe(headers=["Metric", "Value"], label="Error Metrics", interactive=False)
            with gr.Column(scale=2):
                plot_output_forecast = gr.Plot(label="Forecast vs Actual (First 2500 points)")
        plot_btn_forecast.click(plot_forecast_model, inputs=forecast_file, outputs=[metrics_table, plot_output_forecast])

demo.launch()
