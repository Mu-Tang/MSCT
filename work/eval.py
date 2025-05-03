import pandas as pd
import numpy as np
import os

# Load the test result files
file_paths = [
    "./files_new/AblationSingleScale_0.xlsx",
    "./files_new/AblationSingleScale_1.xlsx",
    "./files_new/AblationSingleScale_2.xlsx"
]

# Create eval folder if it doesn't exist
output_dir = "./result_new"
os.makedirs(output_dir, exist_ok=True)

# Define evaluation metrics
def calculate_metrics(preds, actuals):
    n = len(actuals)
    mae = np.mean(np.abs(preds - actuals))
    non_zero_mask = actuals != 0
    mape = np.mean(np.abs((preds[non_zero_mask] - actuals[non_zero_mask]) / actuals[non_zero_mask])) * 100
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))

    # NRMSE in percentage (based on mean of absolute actuals to avoid negative issues)
    mean_actuals = np.mean(np.abs(actuals))  # Use absolute mean to avoid negative mean
    if mean_actuals == 0:
        nrmse = np.nan  # Avoid division by zero
    else:
        nrmse = (rmse / mean_actuals) * 100

    # HAPE calculation: min and max absolute percentage error
    percentage_errors = np.abs((preds - actuals) / actuals) * 100
    min_hape = np.min(percentage_errors)
    max_hape = np.max(percentage_errors)
    hape = [round(min_hape, 6), round(max_hape, 6)]  # Range of HAPE

    # R? calculation
    ss_total = np.sum((actuals - np.mean(actuals)) ** 2)
    ss_residual = np.sum((actuals - preds) ** 2)
    if ss_total == 0:
        r2 = 1.0  # If there's no variance in actuals, assume perfect fit
    else:
        r2 = 1 - (ss_residual / ss_total)

    return {
        "MAE (N)": round(mae, 6), # 平均绝对误差
        "MAPE (%)": round(mape, 6), # 平均相对误差
        "NRMSE (%)": round(nrmse, 6), # 标准化均方根误差
        "HAPE (%)": hape, # 局部最大误差
        "R^2": round(r2, 6) # 决定系数
    }


# Process each file and calculate metrics
all_results = []
for idx, file_path in enumerate(file_paths):
    data = pd.read_excel(file_path, header=None)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    preds, actuals = data[0].values, data[1].values
    metrics = calculate_metrics(preds, actuals)
    metrics["Direction"] = f" {['X', 'Y', 'Z'][idx]}"
    all_results.append(metrics)

# Save results to an Excel file
results_df = pd.DataFrame(all_results)
output_file = os.path.join(output_dir, "evaluation_ASS.xlsx")
results_df.to_excel(output_file, index=False)

# Display the metrics
metrics_df = pd.DataFrame(all_results)
print(metrics_df)

