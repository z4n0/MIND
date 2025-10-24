# =============================================================================
# FIGURE 1: COMBINED VIOLIN + BOX + SWARM PLOTS 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_violin_plots(df, age_col, duration_col, diagnosis_col, colors_dict):
    """
    Create combined violin plots showing age and disease duration distributions.
    
    This is the most informative visualization showing:
    - Distribution shape (violin)
    - Quartiles and median (box)
    - Individual data points (swarm)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- LEFT: Age Distribution ---
    ax = axes[0]
    
    diagnoses_order = ['MSA-P', 'MSA-C','PD']
    
    # Create violin plot
    parts = ax.violinplot(
        [df[df[diagnosis_col] == diag][age_col].dropna() for diag in diagnoses_order],
        positions=range(len(diagnoses_order)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_dict[diagnoses_order[i]])
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add box plot on top
    bp = ax.boxplot(
        [df[df[diagnosis_col] == diag][age_col].dropna() for diag in diagnoses_order],
        positions=range(len(diagnoses_order)),
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )
    
    # Add individual points
    for i, diag in enumerate(diagnoses_order):
        data = df[df[diagnosis_col] == diag][age_col].dropna()
        y = data.values
        x = np.random.normal(i, 0.04, size=len(y))  # Add jitter
        ax.scatter(x, y, alpha=0.6, s=40, color=colors_dict[diag], 
                  edgecolors='black', linewidth=0.5, zorder=3)
    
    # Customize
    ax.set_xticks(range(len(diagnoses_order)))
    ax.set_xticklabels(diagnoses_order, fontsize=12, fontweight='bold')
    ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_title('Age distribution by diagnosis', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add mean ± SD annotations
    for i, diag in enumerate(diagnoses_order):
        data = df[df[diagnosis_col] == diag][age_col].dropna()
        mean_val = data.mean()
        std_val = data.std()
        n = len(data)
        ax.text(i, ax.get_ylim()[1] * 0.98, 
               f'n={n}\n{mean_val:.1f}±{std_val:.1f}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.8))
    
    # --- RIGHT: Disease Duration Distribution ---
    ax = axes[1]
    
    # Create violin plot
    parts = ax.violinplot(
        [df[df[diagnosis_col] == diag][duration_col].dropna() for diag in diagnoses_order],
        positions=range(len(diagnoses_order)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_dict[diagnoses_order[i]])
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add box plot on top
    bp = ax.boxplot(
        [df[df[diagnosis_col] == diag][duration_col].dropna() for diag in diagnoses_order],
        positions=range(len(diagnoses_order)),
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )
    
    # Add individual points
    for i, diag in enumerate(diagnoses_order):
        data = df[df[diagnosis_col] == diag][duration_col].dropna()
        y = data.values
        x = np.random.normal(i, 0.04, size=len(y))  # Add jitter
        ax.scatter(x, y, alpha=0.6, s=40, color=colors_dict[diag], 
                  edgecolors='black', linewidth=0.5, zorder=3)
    
    # Customize
    ax.set_xticks(range(len(diagnoses_order)))
    ax.set_xticklabels(diagnoses_order, fontsize=12, fontweight='bold')
    ax.set_ylabel('Disease duration (years)', fontsize=12, fontweight='bold')
    ax.set_title('Disease duration by diagnosis', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add mean ± SD annotations
    for i, diag in enumerate(diagnoses_order):
        data = df[df[diagnosis_col] == diag][duration_col].dropna()
        mean_val = data.mean()
        std_val = data.std()
        n = len(data)
        ax.text(i, ax.get_ylim()[1] * 0.98, 
               f'n={n}\n{mean_val:.1f}±{std_val:.1f}',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    return fig

# =============================================================================
# FIGURE 2: HISTOGRAMS WITH KDE OVERLAY
# =============================================================================
def create_histogram_plots(df, age_col, duration_col, diagnosis_col, colors_dict):
    """
    Create histograms with KDE overlay for age and disease duration.
    Shows the distribution shape clearly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    diagnoses_order = ['MSA-P', 'MSA-C','PD']
    # --- LEFT: Age Distribution ---
    ax = axes[0]
    for diag in diagnoses_order:
        data = df[df[diagnosis_col] == diag][age_col].dropna()
        ax.hist(data, bins=10, alpha=0.4, label=f'{diag} (n={len(data)})',
               color=colors_dict[diag], edgecolor='black', linewidth=1.5)
        
        # Add KDE
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = kde(x_range) * len(data) * (data.max() - data.min()) / 10
            ax.plot(x_range, kde_values, color=colors_dict[diag], linewidth=2.5)
    
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Age distribution by diagnosis', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # --- RIGHT: Disease Duration Distribution ---
    ax = axes[1]
    for diag in diagnoses_order:
        data = df[df[diagnosis_col] == diag][duration_col].dropna()
        ax.hist(data, bins=10, alpha=0.4, label=f'{diag} (n={len(data)})',
               color=colors_dict[diag], edgecolor='black', linewidth=1.5)
        
        # Add KDE
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = kde(x_range) * len(data) * (data.max() - data.min()) / 10
            ax.plot(x_range, kde_values, color=colors_dict[diag], linewidth=2.5)
    
    ax.set_xlabel('Disease duration (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Disease duration by diagnosis', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def detect_outliers_tukey(df, columns, k=1.5, by_group=None):
    """
    Detect outliers using Tukey's fences (IQR method).
    
    Tukey's fences define outliers as values outside:
    - Lower fence = Q1 - k × IQR
    - Upper fence = Q3 + k × IQR
    
    Where IQR = Q3 - Q1 (interquartile range).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str
        Column names to check for outliers
    k : float, default=1.5
        Multiplier for IQR. Standard values:
        - 1.5: mild outliers (default, Tukey's original)
        - 3.0: extreme outliers
    by_group : str, optional
        Column name to group by (e.g., 'diagnosi_definita').
        If provided, outliers are detected within each group.
    
    Returns
    -------
    outlier_summary : pd.DataFrame
        Summary table with outlier statistics per variable
    outlier_details : dict
        Dictionary mapping column names to DataFrames containing
        outlier rows and metadata
    
    Examples
    --------
    >>> summary, details = detect_outliers_tukey(
    ...     df, 
    ...     ['eta_attuale', 'durata_malattia'],
    ...     by_group='diagnosi_definita'
    ... )
    """
    import numpy as np
    import pandas as pd

    # -- OUTLIER DETECTION: OUTPUTS EXPLAINED --
    # outlier_summary: a list of dictionaries (one per variable/group) with
    #   summary statistics for Tukey outlier detection (Q1, Q3, IQR, boundaries,
    #   counts, etc), returned as a DataFrame for downstream tabular analysis.
    
    # outlier_details: a dict mapping each analyzed column name to a DataFrame 
    #   of the rows in df containing outlier values for that column (either
    #   across the whole sample or per group if by_group is specified), with 
    #   an additional 'outlier_in_column' column specifying the originating field.
    # This facilitates further review, export, or traceability of individual 
    #   outlier patient records for each biomarker/feature.
    
    outlier_summary = [] 
    outlier_details = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"  Column '{col}' not found, skipping.")
            continue
        
        # Get numeric data, drop NaN
        data = df[col].dropna()
        
        if len(data) == 0:
            print(f" Column '{col}' has no valid data, skipping.")
            continue
        
        if by_group and by_group in df.columns:
            # Detect outliers per group (each diagnosis, etc)
            groups = df[by_group].unique()
            groups = groups[pd.notna(groups)] # remove Na values
                
            all_outlier_indices = []
            group_stats = []
            
            for group in groups:
                group_data = df[df[by_group] == group][col].dropna()
                
                if len(group_data) < 4:  # Need at least 4 points for IQR
                    continue
                
                Q1 = group_data.quantile(0.25)
                Q3 = group_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_fence = Q1 - k * IQR
                upper_fence = Q3 + k * IQR
                
                # Find outliers in this group
                group_outlier_mask = (
                    (group_data < lower_fence) | (group_data > upper_fence)
                )
                outlier_indices = group_data[group_outlier_mask].index
                all_outlier_indices.extend(outlier_indices)
                
                n_outliers = len(outlier_indices)
                if n_outliers > 0:
                    group_stats.append({
                        'Variable': col,
                        'Group': group,
                        'N': len(group_data),
                        'Q1': Q1,
                        'Q3': Q3,
                        'IQR': IQR,
                        'Lower_Fence': lower_fence,
                        'Upper_Fence': upper_fence,
                        'N_Outliers': n_outliers,
                        'Outlier_%': (n_outliers / len(group_data)) * 100
                    })
            
            # Combine all groups
            if all_outlier_indices:
                # DataFrame of all group outliers for this col, with a marker column
                outlier_details[col] = df.loc[all_outlier_indices, :].copy()
                outlier_details[col]['outlier_in_column'] = col
                outlier_summary.extend(group_stats)
            
        else: # Detect outliers globally (no grouping)
            # Detect outliers globally (no grouping)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_fence = Q1 - k * IQR
            upper_fence = Q3 + k * IQR
            
            # Find outliers
            outlier_mask = (data < lower_fence) | (data > upper_fence)
            outlier_indices = data[outlier_mask].index
            n_outliers = len(outlier_indices)
            
            outlier_summary.append({
                'Variable': col,
                'Group': 'All',
                'N': len(data),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Lower_Fence': lower_fence,
                'Upper_Fence': upper_fence,
                'N_Outliers': n_outliers,
                'Outlier_%': (n_outliers / len(data)) * 100
            })
            
            if n_outliers > 0:
                outlier_details[col] = df.loc[outlier_indices, :].copy()
                outlier_details[col]['outlier_in_column'] = col
    
    summary_df = pd.DataFrame(outlier_summary)
    
    return summary_df, outlier_details

