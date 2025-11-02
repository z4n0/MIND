import pandas as pd
import numpy as np
# Import plotting libraries and StyleManager and create the output directory to store images
#--------------------------------------------
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
# Add parent directory to path to import style module
# Try adding parent directory if style module not found
try:
    from style.style_manager import StyleManager
except ImportError:
    # If import fails, add parent directory to path
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'notebooks':
        project_root = os.path.dirname(current_dir)
        sys.path.insert(0, project_root)
    else:
        # Try going up one level
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
    from style.style_manager import StyleManager


# Determine project root and initialize style with correct paths
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'notebooks':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

mplstyle_path = os.path.join(project_root, 'style', 'thesis_style.mplstyle')
sm = StyleManager(mplstyle_path=mplstyle_path)
sm.activate()

# Create output directory if it doesn't exist
output_dir = "images/image_analysis"
os.makedirs(output_dir, exist_ok=True)

def _add_full_datetime(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    """
    Internal helper function to create a sortable datetime column.
    
    WARNING: Assumes 'date_col' is in 'mm-dd' format and will use the
    current year for conversion.
    """
    working_df = df.copy()
    datetime_str = working_df[date_col] + ' ' + working_df[time_col]
    
    working_df['full_datetime'] = pd.to_datetime(datetime_str, 
                                                 format='%m-%d %H:%M:%S', 
                                                 errors='coerce')

    if working_df['full_datetime'].isnull().any():
        dropped_count = working_df['full_datetime'].isnull().sum()
        print(f"Warning: Dropped {dropped_count} rows due to unparseable date/time strings.")
        working_df = working_df.dropna(subset=['full_datetime'])
        
    return working_df


def pick_newest_run_per_arch(df: pd.DataFrame, 
                             arch_name_list: list[str],
                             arch_name_col: str = 'model_name', 
                             date_col: str = 'creation_date', 
                             time_col: str = 'creation_time',
                             is_pretrained: bool = False) -> pd.DataFrame:
    """
    Filters for a list of architectures (CASE-INSENSITIVE) and
    selects the single newest run for EACH.

    For non-pretrained: uniqueness is based on model_name only.
    For pretrained: uniqueness is based on (model_name, pretrained_weights) combination.
    
    Treats 'ModelA' and 'modela' as the same architecture.

    Args:
        df (pd.DataFrame): The input DataFrame.
        arch_name_list (list[str]): The list of architecture names to filter for.
        arch_name_col (str): The name of the architecture column.
        date_col (str): The name of the date column.
        time_col (str): The name of the time column.
        is_pretrained (bool): If True, group by both model_name and pretrained_weights.
                             If False, group by model_name only. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing only the single newest run
                      for each architecture (and pretrained_weights if is_pretrained=True)
                      specified in the list.
    """
    
    # 1. Create a lowercase version of the target list for comparison.
    lower_arch_list = [name.lower() for name in arch_name_list]
    
    # 2. Create a working copy and a temporary lowercase column.
    #    This column will be used for filtering and grouping.
    working_df = df.copy()
    working_df['_temp_lower_arch'] = working_df[arch_name_col].str.lower()

    # 3. Filter using the lowercase list and lowercase column.
    filtered_df = working_df[working_df['_temp_lower_arch'].isin(lower_arch_list)]
    
    if filtered_df.empty:
        print(f"No data found for any architectures in the list: {arch_name_list}")
        return pd.DataFrame(columns=df.columns) # Return empty DF

    # 4. Use the helper function to create the 'full_datetime' column.
    with_datetime_df = _add_full_datetime(filtered_df, date_col, time_col)

    if with_datetime_df.empty:
        print(f"No valid datetime data found for the filtered architectures.")
        return pd.DataFrame(columns=df.columns)

    # 5. Sort the DataFrame by the new datetime column (newest first).
    sorted_df = with_datetime_df.sort_values(by='full_datetime', ascending=False)
    
    # 6. Group by the appropriate columns based on is_pretrained flag.
    if is_pretrained:
        # For pretrained: group by both model_name (lowercase) and pretrained_weights
        if 'pretrained_weights' not in sorted_df.columns:
            raise ValueError(
                "is_pretrained=True but 'pretrained_weights' column not found in DataFrame"
            )
        # Keep the newest run for each (model_name, pretrained_weights) combination
        newest_runs_df = sorted_df.drop_duplicates(
            subset=['_temp_lower_arch', 'pretrained_weights'], 
            keep='first'
        )
    else:
        # For non-pretrained: group by model_name (lowercase) only
        newest_runs_df = sorted_df.drop_duplicates(subset=['_temp_lower_arch'], keep='first')
    
    # 7. Return the final DataFrame, dropping the temporary helper columns.
    return newest_runs_df.drop(columns=['full_datetime', '_temp_lower_arch'])



from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_metrics_for_comparison(
    df_ds1: pd.DataFrame,
    df_ds2: pd.DataFrame,
    task_name: str,
    metrics: Sequence[Dict[str, str]],
    model_col: str = "model_name",
) -> Dict[str, dict]:
    """
    Align DS1 and DS2 by architecture and extract mean/std columns
    without any aggregation.

    Each item in `metrics` must have:
      - 'key'     : unique id (e.g., 'patient_major_bal_acc')
      - 'col'     : mean column name (e.g., 'mean_patient_major_bal_acc')
      - 'std_col' : std column name  (e.g., 'std_patient_major_bal_acc')
      - 'label'   : pretty label for plots

    Parameters
    ----------
    df_ds1 : pd.DataFrame
        First experiment's runs (rows = models/runs, columns = metrics and metadata).
    df_ds2 : pd.DataFrame
        Second experiment's runs (same structure as df_ds1).
    task_name : str
        Name of the overall comparison task (used for tracking/plot titles).
    metrics : Sequence[Dict[str, str]]
        List describing which mean/std columns to extract. Each dict must provide
        the keys: 'key', 'col', 'std_col', 'label'. See usage example below.
    model_col : str, default = "model_name"
        The column name containing the architecture/model label to align by.

    Returns
    -------
    out : Dict[str, dict]
        {
            "task": task_name,
            "models": [...],    # list of aligned model names
            "metrics": {
                "<key>": {
                    "label": <label>,
                    "col": <mean_col>,
                    "std_col": <std_col>,
                    "models": [...],  # filtered/valid aligned models
                    "ds1_mean": [...],
                    "ds2_mean": [...],
                    "ds1_std": [...],
                    "ds2_std": [...],
                },
                ...
            }
        }

    Examples
    --------
    >>> metrics = [
    ...     {
    ...         "key": "patient_major_bal_acc",
    ...         "col": "mean_patient_major_bal_acc",
    ...         "std_col": "std_patient_major_bal_acc",
    ...         "label": "Patient Major Balanced Accuracy"
    ...     },
    ...     {
    ...         "key": "patient_major_auc",
    ...         "col": "mean_patient_major_auc",
    ...         "std_col": "std_patient_major_auc",
    ...         "label": "Patient Major AUC"
    ...     }
    ... ]
    >>> results = extract_metrics_for_comparison(df1, df2, "PD vs MSA", metrics)
    >>> print(results["metrics"]["patient_major_bal_acc"].keys())
    dict_keys(['label', 'col', 'std_col', 'models', 'ds1_mean', 'ds2_mean', 'ds1_std', 'ds2_std'])
    >>> print(results["models"])
    ['resnet18_custom', 'efficientnetb1', 'vit3d_msa_thesis']

    Notes
    -----
    - Will raise if required columns are missing.
    - Only aligned models with valid (finite) values for selected metrics are returned.
    """

    if model_col not in df_ds1.columns or model_col not in df_ds2.columns:
        raise KeyError(
            f"`{model_col}` must be present in both dataframes to align models."
        )

    common_models = (
        set(df_ds1[model_col].astype(str).unique())
        & set(df_ds2[model_col].astype(str).unique())
    )
    if not common_models:
        raise ValueError("No common models found between DS1 and DS2.")
    common_models = sorted(common_models)

    ds1_unique = (
        df_ds1[df_ds1[model_col].astype(str).isin(common_models)]
        .drop_duplicates(subset=[model_col])
        .set_index(model_col)
        .loc[common_models]
    )
    
    ds2_unique = (
        df_ds2[df_ds2[model_col].astype(str).isin(common_models)]
        .drop_duplicates(subset=[model_col])
        .set_index(model_col)
        .loc[common_models]
    )

    out: Dict[str, dict] = {
        "task": task_name,
        "models": common_models,
        "metrics": {},
    }

    for m in metrics:
        key = m["key"]
        col = m["col"]
        std_col = m["std_col"]
        label = m["label"]

        for need in (col, std_col):
            if need not in ds1_unique.columns:
                raise KeyError(f"Column '{need}' not found in DS1.")
            if need not in ds2_unique.columns:
                raise KeyError(f"Column '{need}' not found in DS2.")

        s1_mean = ds1_unique[col]
        s2_mean = ds2_unique[col]
        s1_std = ds1_unique[std_col]
        s2_std = ds2_unique[std_col]

        # Valid if all four values are finite for that model
        valid_mask = ~(
            s1_mean.isna() | s2_mean.isna() | s1_std.isna() | s2_std.isna()
        )

        aligned_models = list(np.array(common_models)[valid_mask.values])

        out["metrics"][key] = {
            "label": label,
            "col": col,
            "std_col": std_col,
            "models": aligned_models,
            "ds1_mean": s1_mean[valid_mask].astype(float).tolist(),
            "ds2_mean": s2_mean[valid_mask].astype(float).tolist(),
            "ds1_std": s1_std[valid_mask].astype(float).tolist(),
            "ds2_std": s2_std[valid_mask].astype(float).tolist(),
        }

    return out


def create_comparison_figure(
    results: Dict[str, dict],
    metric_key: str,
    title: str,
    filename: str,
    ds1_label: str = "DS1",
    ds2_label: str = "DS2",
    ylim: Tuple[float, float] | None = (0.0, 1.0),
) -> str:
    """
    Grouped bar plot (DS1 vs DS2) per-architecture for a single metric,
    with error bars from per-dataset std columns.
    """
    if metric_key not in results["metrics"]:
        raise KeyError(
            f"Metric '{metric_key}' not found. Available: "
            f"{list(results['metrics'].keys())}"
        )

    block = results["metrics"][metric_key]
    models = block["models"]
    ds1_vals = block["ds1_mean"]
    ds2_vals = block["ds2_mean"]
    ds1_err = block["ds1_std"]
    ds2_err = block["ds2_std"]
    y_label = block["label"]

    if not models:
        raise ValueError(
            f"No aligned models with valid values for metric '{metric_key}'."
        )

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(0.85 * max(6, len(models)), 4.5))

    bars1 = ax.bar(
        x - width / 2,
        ds1_vals,
        width,
        yerr=ds1_err,
        ecolor="black",
        capsize=4,
        label=ds1_label,
        color=sm.palette[7],
    )
    bars2 = ax.bar(
        x + width / 2,
        ds2_vals,
        width,
        yerr=ds2_err,
        ecolor="black",
        capsize=4,
        label=ds2_label,
        color=sm.palette[6],
    )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(frameon=True, loc="best", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bar heights (means)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = os.path.join("images/image_analysis", filename)
    sm.savefig(output_path)
    print(f"Figure saved to {output_path}")
    # plt.show()
    return output_path