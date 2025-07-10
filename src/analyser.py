import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import re

# --- Robust Parsing Functions (v2 - Confirmed Working) ---

def parse_loss_history_v2(file_path):
    """Robustly parses the loss.dat file by reading line-by-line."""
    epochs, total_losses = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = [p for p in line.strip().split() if p]
            if len(parts) >= 2:
                try:
                    epoch = int(float(parts[0]))
                    total_loss = float(parts[1])
                    epochs.append(epoch)
                    total_losses.append(total_loss)
                except (ValueError, IndexError):
                    continue
    if not epochs:
        raise ValueError(f"No valid epoch/loss data found in {file_path}")
    return pd.Series(total_losses, index=epochs, name="Total Loss")

def parse_variables_log(file_path):
    """Parses the variables.dat log file into a DataFrame."""
    epochs, param_values = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split('[')
            if len(parts) < 2: continue
            epoch_str = parts[0].strip()
            if not epoch_str: continue
            epoch = int(float(epoch_str))
            numbers_str = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*", parts[1])
            if numbers_str:
                epochs.append(epoch)
                param_values.append([float(n) for n in numbers_str])
    if not epochs:
        raise ValueError(f"Could not parse epoch/parameter log from {file_path}")
    return pd.DataFrame(param_values, index=epochs)

# --- Main Analysis and Reporting Script ---

def generate_final_report(base_path=".", save_outputs=True):
    """
    Performs full analysis, generates plots, and saves tables mimicking the research paper.
    """
    print("--- ROMPINN Final Report & Table Generator ---")
    
    results_dir = Path(base_path) / "results" / "models"
    output_dir = Path(base_path) / "paper"
    if save_outputs:
        output_dir.mkdir(exist_ok=True)
        print(f"Outputs will be saved to: {output_dir.resolve()}")

    model_data = []
    
    # --- Data Extraction ---
    print("\nProcessing model output files...")
    for i in range(1, 9):
        model_name = f"model{i}"
        model_path = results_dir / model_name
        if not model_path.is_dir(): continue

        try:
            loss_history_series = parse_loss_history_v2(model_path / "loss.dat")
            param_log_df = parse_variables_log(model_path / "variables.dat")
            final_params_series = param_log_df.iloc[-1]
            
            model_data.append({
                "Model_Number": i,
                "Model_Name": model_name,
                "Loss_History": loss_history_series,
                "Param_Log": param_log_df,
                "Final_Params": final_params_series.to_dict(),
                "Final_MSE": loss_history_series.iloc[-1] # Using final total loss as a proxy
            })
            print(f"  - Successfully processed {model_name}")
        except Exception as e:
            print(f"  - Error processing {model_name}: {e}")

    if not model_data:
        print("\nNo model data loaded. Exiting.")
        return

    # --- Generate Figure 5: Loss History ---
    print("\nGenerating Figure 5: Loss History Plot...")
    fig5, axes5 = plt.subplots(4, 2, figsize=(15, 20))
    axes5 = axes5.flatten()
    sns.set_style("whitegrid")
    # ... (Plotting code for Figure 5 is unchanged) ...
    for idx, data in enumerate(model_data):
        ax = axes5[idx]
        loss_history = data["Loss_History"]
        sns.lineplot(x=loss_history.index, y=loss_history.values, ax=ax, color='darkorange')
        ax.set_title(f"({chr(97+idx)}) Model {data['Model_Number']}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
    for i in range(len(model_data), len(axes5)): axes5[i].set_visible(False)
    plt.tight_layout(pad=3.0)
    fig5.suptitle("Figure 5: Cumulative Training Loss", fontsize=16, y=1.02)
    if save_outputs:
        fig5_path = output_dir / "Figure5_Loss_History.png"
        plt.savefig(fig5_path, dpi=300)
        print(f"  - Figure 5 saved to {fig5_path}")
    plt.show()

    # --- Generate Figure 6: Parameter Evolution ---
    print("\nGenerating Figure 6: Parameter Evolution Plot...")
    # ... (Plotting code for Figure 6 is unchanged) ...
    param_map = {
        1: ['α1', 'β1'], 2: ['α1', 'α2', 'β1', 'β2'], 3: ['α1', 'α2', 'β1', 'β2'],
        4: ['α1', 'α2', 'α3', 'β1', 'β2', 'β3'], 5: ['α1', 'α2', 'α3', 'β1', 'β2', 'β3'],
        6: ['α1', 'α2', 'α3', 'β1', 'β2', 'β3'], 7: ['α1', 'α2', 'α3', 'β1', 'β2', 'β3'],
        8: ['α1', 'α2', 'α3', 'α4', 'β1', 'β2', 'β3', 'β4']
    }
    fig6, axes6 = plt.subplots(4, 2, figsize=(15, 20))
    axes6 = axes6.flatten()
    for idx, data in enumerate(model_data):
        ax = axes6[idx]
        log_df = data["Param_Log"]
        model_num = data["Model_Number"]
        if model_num in param_map:
            log_df.columns = param_map[model_num][:len(log_df.columns)]
        for col in log_df.columns:
            sns.lineplot(x=log_df.index, y=log_df[col], ax=ax, label=col)
        ax.set_title(f"({chr(97+idx)}) Model {model_num}", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rate Parameter Value")
        ax.legend(title="Parameters")
        ax.tick_params(axis='x', rotation=45)
    for i in range(len(model_data), len(axes6)): axes6[i].set_visible(False)
    plt.tight_layout(pad=3.0)
    fig6.suptitle("Figure 6: Learned Rate Parameter Evolution", fontsize=16, y=1.02)
    if save_outputs:
        fig6_path = output_dir / "Figure6_Parameter_Evolution.png"
        plt.savefig(fig6_path, dpi=300)
        print(f"  - Figure 6 saved to {fig6_path}")
    plt.show()


    # --- Generate Table 3 (Console) and Table 5 (Console and CSV) ---
    
    # Table 3: Goodness of Fit
    print("\n--- [ Table 3: Goodness of Fit (ROMPINN Section) ] ---\n")
    fit_df = pd.DataFrame([{
        "Model": data["Model_Name"],
        "Final Total Loss": f"{data['Final_MSE']:.6f}"
    } for data in model_data])
    print(fit_df.to_string(index=False))

    # Table 5: Final Parameter Values
    print("\n--- [ Table 5: Final Estimated Parameters ] ---\n")
    all_param_names = [f'α{i}' for i in range(1, 5)] + [f'β{i}' for i in range(1, 5)]
    table_rows = []

    for data in model_data:
        row = {"Model": data["Model_Name"]}
        final_params = data["Final_Params"]
        param_names_for_model = param_map.get(data["Model_Number"], [])
        
        for i, param_name in enumerate(param_names_for_model):
            row[param_name] = final_params.get(i, np.nan)
            
        table_rows.append(row)
        
    params_df = pd.DataFrame(table_rows)
    # Reorder columns to match the paper's table for consistent output
    final_columns = ["Model"] + [p for p in all_param_names if p in params_df.columns]
    params_df = params_df[final_columns]
    
    # Create a version for printing (with 'X' and formatting)
    params_df_print = params_df.copy()
    params_df_print = params_df_print.fillna("X")
    for col in params_df_print.columns:
        if col != "Model":
            params_df_print[col] = params_df_print[col].apply(lambda x: f"{x:.5f}" if isinstance(x, float) else x)
            
    print(params_df.to_string(index=False))

    if save_outputs:
        # Save the raw numerical data to CSV, keeping NaNs for empty cells
        csv_path = output_dir / "Table5_Final_Parameters.csv"
        params_df.to_csv(csv_path, index=False, float_format='%.5f')
        print(f"\n  - Table 5 saved to {csv_path}")
    
    print("\n\n--- Report Generation Complete ---")


if __name__ == '__main__':
    project_root_path = "."
    generate_final_report(project_root_path, save_outputs=True)