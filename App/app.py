import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import re
from datetime import datetime
import sys
import nbformat
import ast
import types

# =============================================================================
# ------------- Load base dataset for the other tabs -------------
# =============================================================================
BASE_PATH = "../WindPowerForecastingData.xlsx"
if os.path.exists(BASE_PATH):
    df = pd.read_excel(BASE_PATH)
else:
    # Fallback empty df to keep the UI alive if the file isn't present
    df = pd.DataFrame(columns=["TIMESTAMP", "TARGETVAR", "U10", "V10", "U100", "V100"])

if "TIMESTAMP" in df.columns:
    # NOTE: Parsing here is only for UI/plots; we do NOT save parsed datetime to disk.
    try:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d %H:%M")
    except Exception:
        try:
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        except Exception:
            pass

# =============================================================================
# ------------- Helper functions (Overview / Viz / Stats / Forecast) -------------
# =============================================================================
def data_overview():
    shape_info = f"Data Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    time_info = ""
    if "TIMESTAMP" in df.columns and pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]):
        try:
            time_info = f"Time Range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}"
        except Exception:
            time_info = ""
    return df, f"{shape_info}\n{time_info}"

def plot_data(x_col, y_col, plot_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    if x_col is None or y_col is None:
        ax.set_title("Please select both X and Y columns")
        ax.axis("off")
        return fig

    if plot_type == "Line":
        ax.plot(df[x_col], df[y_col])
    elif plot_type == "Scatter":
        ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True)
    return fig

def get_forecast_file(model_name):
    if model_name == "BiLSTM":
        return "../data_logs/forecast_results_bilstm.csv"
    elif model_name == "BiGLSTM":
        return "../data_logs/forecast_results_biglstm.csv"
    else:
        return None

def compute_error_metrics_table(df_fore):
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

def show_stats_all(column):
    if column is None:
        empty = pd.DataFrame({"Statistic": [], "Value": []})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Please choose a numeric column")
        ax.axis("off")
        return empty, fig

    stats = df[column].describe().to_frame().reset_index()
    stats.columns = ["Statistic", "Value"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df[column].hist(ax=ax[0], bins=30)
    ax[0].set_title(f"Histogram of {column}")
    df.boxplot(column=column, ax=ax[1])
    ax[1].set_title(f"Boxplot of {column}")
    plt.tight_layout()
    return stats, fig

def plot_forecast(filename, plot_len=2500):
    df_full = pd.read_csv(filename)
    plot_df = df_full.iloc[:min(plot_len, len(df_full))].reset_index(drop=True)

    metrics_df = compute_error_metrics_table(df_full)
    acc = metrics_df.loc[metrics_df["Metric"] == "R² Score", "Value"].values[0]

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
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

def plot_forecast_model(model_name):
    filename = get_forecast_file(model_name)
    if filename is None or not os.path.exists(filename):
        return pd.DataFrame({"Error": ["Unknown model selected or file not found."]}), None
    df_full = pd.read_csv(filename)
    metrics_df = compute_error_metrics_table(df_full)
    fig = plot_forecast(filename, plot_len=2500)
    return metrics_df, fig

# =============================================================================
# ------------- Strict-but-friendly file validation for Generation Prediction -------------
# =============================================================================
def _normalize(colname: str) -> str:
    s = str(colname).lower().strip()
    return "".join(ch for ch in s if ch.isalnum())

def _read_any_table(path: str, sheet_name=0) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls")

def _try_promote_header(df_in: pd.DataFrame, required_norm: set, max_scan_rows: int = 10) -> pd.DataFrame:
    """
    If the real header is inside the first few rows, promote that row to header.
    Works when files have title rows and then the actual header on row 2..N.
    """
    df = df_in.copy()
    looks_unnamed = all(
        str(c).lower().startswith("unnamed") or isinstance(c, (int, np.integer))
        for c in df.columns
    )
    if not looks_unnamed:
        return df

    scan = min(max_scan_rows, len(df))
    for i in range(scan):
        row_vals = [_normalize(v) for v in df.iloc[i].tolist()]
        row_set = set(row_vals)
        if required_norm.issubset(row_set):
            df2 = df.iloc[i+1:].copy()
            df2.columns = df.iloc[i].astype(str).tolist()
            df2.reset_index(drop=True, inplace=True)
            return df2
    return df

def parse_wind_input_table(file_path: str, sheet_name=0) -> pd.DataFrame:
    """
    Accept columns (at least): TIMESTAMP, TARGETVAR, U10, V10, U100, V100.
    Extra columns are allowed and will be ignored.
    """
    df_raw = _read_any_table(file_path, sheet_name=sheet_name)
    if df_raw is None or df_raw.empty:
        raise ValueError("Uploaded file is empty or unreadable.")

    required = ["TIMESTAMP", "TARGETVAR", "U10", "V10", "U100", "V100"]
    required_norm = {_normalize(c) for c in required}

    # Try to promote a header row if the current header looks wrong
    df_raw = _try_promote_header(df_raw, required_norm=required_norm, max_scan_rows=10)

    # Build normalization map (last occurrence wins if duplicates)
    norm_map = {}
    for c in df_raw.columns:
        nc = _normalize(c)
        if nc:
            norm_map[nc] = c
    normalized_cols = set(norm_map.keys())

    # Missing requirement check
    missing_norm = required_norm - normalized_cols
    if missing_norm:
        missing_readable = [r for r in required if _normalize(r) in missing_norm]
        found_cols = [str(c) for c in df_raw.columns]
        raise ValueError(
            "Missing required columns: "
            f"{missing_readable}. Found columns: {found_cols}. "
            "Hints: names are case/space/symbol-insensitive (e.g., 'Target Var' is OK)."
        )

    # Select required columns in correct order
    out = df_raw[[norm_map[_normalize(c)] for c in required]].copy()
    out.columns = required

    # TIMESTAMP cleanup and validation — keep as string, do not parse
    out["TIMESTAMP"] = out["TIMESTAMP"].astype(str).str.strip()
    if (out["TIMESTAMP"] == "").any():
        bad_idxs = out.index[out["TIMESTAMP"] == ""].tolist()[:5]
        raise ValueError(f"Empty TIMESTAMP values at rows (0-based): {bad_idxs}. Please fix and re-upload.")

    # Coerce numeric (allows numeric strings)
    for c in ["TARGETVAR", "U10", "V10", "U100", "V100"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Validate numeric presence
    bad_mask = out[["TARGETVAR", "U10", "V10", "U100", "V100"]].isna().any(axis=1)
    if bad_mask.any():
        idxs = out.index[bad_mask].tolist()[:5]
        raise ValueError(
            "Non-numeric or missing values detected in TARGETVAR/U10/V10/U100/V100. "
            f"Example bad row indices (0-based): {idxs}"
        )

    # Ensure final column order
    out = out[["TIMESTAMP", "TARGETVAR", "U10", "V10", "U100", "V100"]]
    return out

# ---------- NEW: parser for features-only file (U10, V10, U100, V100) ----------
def parse_features_table(file_path: str, sheet_name=0) -> pd.DataFrame:
    """
    Accepts ONLY: U10, V10, U100, V100 (numeric). Extra columns are ignored.
    Returns DataFrame with exactly these columns in this order.
    """
    df_raw = _read_any_table(file_path, sheet_name=sheet_name)
    if df_raw is None or df_raw.empty:
        raise ValueError("Features file is empty or unreadable.")

    required = ["U10", "V10", "U100", "V100"]
    required_norm = {_normalize(c) for c in required}

    df_raw = _try_promote_header(df_raw, required_norm=required_norm, max_scan_rows=10)

    # Build normalization map
    norm_map = {}
    for c in df_raw.columns:
        nc = _normalize(c)
        if nc:
            norm_map[nc] = c
    normalized_cols = set(norm_map.keys())

    missing_norm = required_norm - normalized_cols
    if missing_norm:
        missing_readable = [r for r in required if _normalize(r) in missing_norm]
        found_cols = [str(c) for c in df_raw.columns]
        raise ValueError(
            "Features file missing required columns: "
            f"{missing_readable}. Found columns: {found_cols}."
        )

    out = df_raw[[norm_map[_normalize(c)] for c in required]].copy()
    out.columns = required

    # Coerce numeric
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    bad_mask = out[required].isna().any(axis=1)
    if bad_mask.any():
        idxs = out.index[bad_mask].tolist()[:5]
        raise ValueError(
            "Features file contains non-numeric or missing values in U10/V10/U100/V100. "
            f"Example bad row indices (0-based): {idxs}"
        )

    # Ensure final column order
    out = out[["U10", "V10", "U100", "V100"]]
    return out

# =============================================================================
# ------------- Storage config & file save -------------
# =============================================================================
SAVE_DIR = Path(r"../data_logs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_input_dataframe_with_timestamp(df_in: pd.DataFrame) -> Path:
    """
    Save the cleaned input data WITH the TIMESTAMP column to input_data.csv.
    """
    out = df_in.copy()
    out["TIMESTAMP"] = out["TIMESTAMP"].astype(str)
    cols = ["TIMESTAMP", "TARGETVAR", "U10", "V10", "U100", "V100"]
    out = out[cols]
    dest = SAVE_DIR / "input_data.csv"
    out.to_csv(dest, index=False)
    return dest

# ---------- NEW: saver for features-only file ----------
def save_features_dataframe(df_feat: pd.DataFrame) -> Path:
    """
    Save the features-only data to input_features.csv (U10, V10, U100, V100).
    """
    dest = SAVE_DIR / "input_features.csv"
    df_feat[["U10", "V10", "U100", "V100"]].to_csv(dest, index=False)
    return dest

def save_inputs(model_name: str, pred_datetime: str) -> Path:
    """
    OVERWRITE inputs.csv with the latest submission only.
    """
    dest = SAVE_DIR / "inputs.csv"
    df_row = pd.DataFrame([{"chosen_model": model_name, "prediction_datetime": pred_datetime}])
    df_row.to_csv(dest, index=False)  # overwrite
    return dest

# =============================================================================
# ------------- Helpers: validate 'YYYY/MM/DD HH.MM' -------------
# =============================================================================
def _validate_and_extract_hour(dt_str: str) -> float:
    if dt_str is None or str(dt_str).strip() == "":
        raise ValueError("Please enter a prediction date & time in the format YYYY/MM/DD HH.MM (e.g., 2012/01/01 05.00).")

    s = str(dt_str).strip()
    m = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{1,2})\.(\d{2})", s)
    if not m:
        raise ValueError("Invalid format. Use YYYY/MM/DD HH.MM (e.g., 2012/01/01 05.00).")

    yyyy, mm, dd, hh, mins = m.groups()
    try:
        datetime.strptime(f"{yyyy}/{mm}/{dd}", "%Y/%m/%d")
    except ValueError:
        raise ValueError("Invalid date. Please check year/month/day.")
    h = int(hh)
    if not (0 <= h <= 24):
        raise ValueError("Hour must be between 0 and 24.")
    if mins != "00":
        raise ValueError("Minutes must be '00' (hourly steps only).")
    return float(h)

# =============================================================================
# ------------- Dynamic loader for notebook modules (SAFE: only defs) -------------
# =============================================================================

def _extract_safe_module_code_from_notebook(notebook_path: Path) -> str:
    nb_node = nbformat.read(str(notebook_path), as_version=4)
    all_src = []
    for cell in nb_node.cells:
        if cell.cell_type == "code":
            all_src.append(cell.source)
    src = "\n\n".join(all_src)
    tree = ast.parse(src)
    safe_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            safe_nodes.append(node)
    safe_mod = ast.Module(body=safe_nodes, type_ignores=[])
    try:
        safe_code = ast.unparse(safe_mod)  # Py>=3.9
    except Exception:
        compile(safe_mod, "<ast>", "exec")
        safe_code = "pass"
    return safe_code

def _inject_defaults(mod):
    # Put any “missing” globals your notebook’s test_function might expect
    try:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        _device = "cpu"

    defaults = {
        "TIME_COL":   "TIMESTAMP",
        "TARGET_COL": "TARGETVAR",
        "BASE_FEATS": ["U10", "V10", "U100", "V100"],
        "LAGS_Y":     [1, 3, 6, 12, 24],
        "LAGS_SPEED": [1, 3, 6],
        "ROLLS_Y":    [6, 12, 24],
        "TURB_WINS":  [6, 12, 24, 48],
        "TRAIN_RATIO": 0.70,
        "VAL_RATIO":   0.15,
        "DEVICE": _device,
        "DATA_LOGS_DIR": Path("../data_logs"),
        "INPUT_DATA_PATH": Path("../data_logs/input_data.csv"),
        "INPUT_FEATURES_PATH": Path("../data_logs/input_features.csv"),
        "INPUTS_PATH": Path("../data_logs/inputs.csv"),
    }
    for k, v in defaults.items():
        mod.__dict__.setdefault(k, v)  # only set if missing

def load_test_function(model_name: str):
    base_folder = Path(r"../Predictions")
    if model_name == "BiLSTM":
        notebook_path = base_folder / "biLSTM_pred.ipynb"
    elif model_name == "BiGLSTM":
        notebook_path = base_folder / "biGLSTM_pred.ipynb"
    else:
        raise ValueError("Unknown model name. Use 'BiLSTM' or 'BiGLSTM'.")

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    safe_code = _extract_safe_module_code_from_notebook(notebook_path)

    mod = types.ModuleType(f"_safe_nb_{notebook_path.stem}")
    exec(safe_code, mod.__dict__)          # <-- we execute the notebook’s safe code into mod

    _inject_defaults(mod)                  # <-- RIGHT AFTER exec: add missing globals into mod

    if not hasattr(mod, "test_function"):
        raise AttributeError(
            f"'test_function' not found in {notebook_path.name}. "
            "Make sure the notebook defines a function named test_function(...)."
        )
    return getattr(mod, "test_function")



# Single, canonical predict handler used by the UI
def predict_handler(model_name: str) -> str:
    test_fn = load_test_function(model_name)
    y_pred = test_fn()  # your notebook's test_function should read the saved CSVs if needed
    return str(y_pred)

# =============================================================================
# ------------- Submit handler for Generation Prediction -------------
# =============================================================================
def submit_upload(file_main, file_features, model_name, pred_dt_str):
    # Validate model
    if model_name not in ("BiLSTM", "BiGLSTM"):
        return "Validation error: Please choose a forecasting model (BiLSTM or BiGLSTM)."

    # Validate prediction date & time (YYYY/MM/DD HH.MM)
    try:
        _ = _validate_and_extract_hour(pred_dt_str)  # validate only
    except Exception as e:
        return f"Validation error: {e}"

    # Validate file selections
    if file_main is None:
        return ("Validation error: Please upload the MAIN sheet (TIMESTAMP, TARGETVAR, U10, V10, U100, V100).")
    if file_features is None:
        return ("Validation error: Please upload the FEATURES sheet (U10, V10, U100, V100).")

    # Parse & validate uploaded files
    try:
        df_main = parse_wind_input_table(file_main)
    except Exception as e:
        return f"Validation error in MAIN sheet: {e}"

    try:
        df_feat = parse_features_table(file_features)
    except Exception as e:
        return f"Validation error in FEATURES sheet: {e}"

    # Save artifacts (overwrite each time)
    try:
        save_input_dataframe_with_timestamp(df_main)   # writes input_data.csv (with TIMESTAMP)
        save_features_dataframe(df_feat)               # writes input_features.csv (features-only)
        save_inputs(model_name, pred_dt_str)           # writes inputs.csv
    except Exception as e:
        return f"Unexpected save error: {e}"

    # --- Call prediction ---
    try:
        y_pred = predict_handler(model_name)
        return f"Predicted Value: {y_pred}"
    except Exception as e:
        return f"Prediction error: {e}"

# =============================================================================
# ------------- Gradio UI -------------
# =============================================================================
with gr.Blocks() as demo:
    gr.Markdown('<p style="font-size: 2.0em; font-weight: bold;">🌬 Wind Power Forecasting Data Analysis</p>')

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
                x_choices = list(df.columns)
                x_default = x_choices[0] if len(x_choices) > 0 else None
                y_choices = list(df.columns)
                default_y = (y_choices[1] if len(y_choices) > 1
                             else (y_choices[0] if len(y_choices) > 0 else None))
                x_col = gr.Dropdown(choices=x_choices, value=x_default, label="X-axis")
                y_col = gr.Dropdown(choices=y_choices, value=default_y, label="Y-axis")
                plot_type = gr.Radio(choices=["Line", "Scatter"], value="Line", label="Plot Type")
                plot_btn_vis = gr.Button("Plot")
            with gr.Column(scale=2):
                plot_output_vis = gr.Plot(label="Plot")
                plot_btn_vis.click(plot_data, inputs=[x_col, y_col, plot_type], outputs=plot_output_vis)

    with gr.Tab("Statistics"):
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        default_num = numeric_cols[0] if len(numeric_cols) > 0 else None
        col_choice = gr.Dropdown(choices=numeric_cols, value=default_num, label="Select Parameter")
        stats_btn = gr.Button("Show Stats")
        stats_output = gr.Dataframe(label="Summary Statistics")
        stats_plot = gr.Plot(label="Histogram & Boxplot")
        stats_btn.click(show_stats_all, inputs=col_choice, outputs=[stats_output, stats_plot])

    # --- Forecast Analysis Tab ---
    with gr.Tab("Forecast Analysis"):
        gr.Markdown("## Forecasting Model Evaluation")
        with gr.Row():
            with gr.Column(scale=1):
                forecast_file = gr.Dropdown(["BiLSTM", "BiGLSTM"], label="Choose Model")
                plot_btn_forecast = gr.Button("Show Forecast Results")
                metrics_table = gr.Dataframe(label="Error Metrics", interactive=False)
            with gr.Column(scale=2):
                plot_output_forecast = gr.Plot(label="Forecast vs Actual (First 2500 points)")
                plot_btn_forecast.click(plot_forecast_model,
                                        inputs=forecast_file,
                                        outputs=[metrics_table, plot_output_forecast])

        gr.Markdown("")
        gr.Markdown("## Generation Prediction")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gen_time_file_main = gr.File(
                        label="MAIN sheet: TIMESTAMP, TARGETVAR, U10, V10, U100, V100 (.csv/.xlsx/.xls)",
                        file_types=[".csv", ".xlsx", ".xls"],
                        type="filepath"
                    )
                    
                with gr.Group():
                    gen_time_file_features = gr.File(
                        label="WEATHER DATA sheet: U10, V10, U100, V100 (.csv/.xlsx/.xls)",
                        file_types=[".csv", ".xlsx", ".xls"],
                        type="filepath"
                    )
                forecast_file2 = gr.Dropdown(["BiLSTM", "BiGLSTM"], label="Choose Model (Prediction)")
                with gr.Group():
                    prediction_datetime_input = gr.Textbox(
                        label="Prediction Date & Time (YYYY/MM/DD HH.MM)",
                        placeholder="e.g., 2012/01/01 05.00",
                    )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                with gr.Group():
                    output = gr.Textbox(
                        label="Predicted Generation",
                        placeholder="(Prediction result will appear here)",
                        lines=6,
                        interactive=False
                    )

            submit_btn.click(
                submit_upload,
                inputs=[gen_time_file_main, gen_time_file_features, forecast_file2, prediction_datetime_input],
                outputs=output
            )

            # Reset to clean state (including clearing output)
            clear_btn.click(
                lambda: (gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=""), ""),
                inputs=[],
                outputs=[gen_time_file_main, gen_time_file_features, forecast_file2, prediction_datetime_input, output]
            )

demo.launch()
