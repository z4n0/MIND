import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from style.style_manager import StyleManager
import os
import pandas as pd
from pathlib import Path
# Get the absolute path to this notebook's directory
# Get the absolute path to this notebook's directory
NOTEBOOK_DIR = Path.cwd()

# Find the project root by looking for a marker file/directory
PROJ_ROOT = NOTEBOOK_DIR
while PROJ_ROOT != PROJ_ROOT.parent:
    if (PROJ_ROOT / "style").exists() and (PROJ_ROOT / "clinica").exists():
        break
    PROJ_ROOT = PROJ_ROOT.parent

PROJ_ROOT = NOTEBOOK_DIR.parent

STYLE_PATH = PROJ_ROOT / "style" / "thesis_style.mplstyle"
sm = StyleManager(mplstyle_path=str(STYLE_PATH))
sm.activate()


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
                        'variable': col,
                        'group': group,
                        'n': len(group_data),
                        'q1': Q1,
                        'q3': Q3,
                        'iqr': IQR,
                        'lower_fence': lower_fence,
                        'upper_fence': upper_fence,
                        'n_outliers': n_outliers,
                        'outlier_%': (n_outliers / len(group_data)) * 100
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
                'variable': col,
                'group': 'All',
                'n': len(data),
                'q1': Q1,
                'q3': Q3,
                'iqr': IQR,
                'lower_fence': lower_fence,
                'upper_fence': upper_fence,
                'n_outliers': n_outliers,
                'outlier_%': (n_outliers / len(data)) * 100
            })
            
            if n_outliers > 0:
                outlier_details[col] = df.loc[outlier_indices, :].copy()
                outlier_details[col]['outlier_in_column'] = col
    
    summary_df = pd.DataFrame(outlier_summary)
    
    return summary_df, outlier_details



def plot_missing_by_variable(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    show_percentage: bool = True,
    threshold: float = None,
    save_path: str = None,
    diagnosis_name: str = None
) -> plt.Figure:
    """
    Create a bar chart showing the count/percentage of missing values
    per variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze for missing values.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 6).
    show_percentage : bool, optional
        If True, display percentages; if False, display counts.
        Default is True.
    threshold : float, optional
        If provided, draw a horizontal line at this threshold (e.g., 0.05
        for 5% missingness threshold).
    save_path : str, optional
        If provided, save the figure to this path.
        
    Returns
    -------
    fig : plt.Figure
        The created figure object.
        
    Notes
    -----
    Variables are sorted by missingness (highest to lowest) for easier
    interpretation.
    """
    # Initialize style
    sm = StyleManager(mplstyle_path=str(STYLE_PATH))
    sm.activate()
    
    # Calculate missing values
    if show_percentage:
        missing_data = (df.isnull().sum() / len(df) * 100).sort_values(
            ascending=False
        )
        ylabel = "Missing Values (%)"
        value_format = ".1f"
    else:
        missing_data = df.isnull().sum().sort_values(ascending=False)
        ylabel = "Missing Values (Count)"
        value_format = ".0f"
    
    # Filter out variables with no missing values
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        print("No missing values found in the dataframe.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    msa_red_flags = [#AI generated
        'poor_l_dopa_responsivenes', 'cerebellar_syndrome', 
        'rapid_progression_w_3_yrs', 'moderate_to_severe_postural_instability_w_3_yrs_of_motor_onset',
        'severe_speech_impairement_w_3_yrs', 'severe_dysphagia_w_3_yrs', 'stridor'
    ]
    msa_red_flags_clinic_certified = [ #Clinicians certified
        'poor_l_dopa_responsivenes',
        'rapid_progression_w_3_yrs',
        'moderate_to_severe_postural_instability_w_3_yrs_of_motor_onset',
        'craniocervical_dyst_induced_dy_l_dopa',
        'severe_speech_impairement_w_3_yrs',
        'severe_dysphagia_w_3_yrs',
        'unexplained_babinski',
        'jerky_myoclonic_postural_or_kinetic_tremor',
        'postural_deformities',
        'unexplained_voiding_difficulties',
        'unexplained_urinary_urge_incontinence',
        'stridor',
        'inspiratory_sighs',
        'cold_discolored_hands_and_feet',
        'pathologic_laughter_or_crying'
    ]

    colors_dict = {
        'MSA-P': sm.palette[0],
        'MSA-C': sm.palette[1],
        'PD': sm.palette[2],
        'MSA': sm.palette[3]
    }
    if diagnosis_name:
        if diagnosis_name == 'MSA-P':
            color = colors_dict['MSA-P']
        elif diagnosis_name == 'MSA-C':
            color = colors_dict['MSA-C']
        elif diagnosis_name == 'PD':
            color = colors_dict['PD']
        elif diagnosis_name == 'MSA':
            color = colors_dict['MSA']
        else:
            color = colors_dict['MSA']
    # Create bar chart
    bars = ax.bar(
        range(len(missing_data)),
        missing_data.values,
        color=color,  # Use vermillion from Okabe-Ito palette
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5
    )
    
    # Add threshold line if specified
    if threshold is not None:
        threshold_val = threshold if show_percentage else threshold * len(df)
        ax.axhline(
            y=threshold_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.0f}%)" if not show_percentage 
                  else f"Threshold ({threshold})"
        )
        ax.legend(loc="upper right")
    
    # Formatting
    ax.set_xticks(range(len(missing_data)))
    ax.set_xticklabels(
        missing_data.index,
        rotation=45,
        ha="right",
        fontsize=10
    )
    ax.set_xlabel("Variables", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if diagnosis_name:
        ax.set_title(
            f"Missing Values by Variable for {diagnosis_name}",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
    else:
        ax.set_title(
            "Missing Values by Variable",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
    
    # Add value labels on top of bars ONLY if there are 12 or fewer columns
    if len(missing_data) <= 12:
        for i, (bar, value) in enumerate(zip(bars, missing_data.values)):
            ax.text(
                i,
                value + (max(missing_data.values) * 0.01),
                f"{value:{value_format}}",
                ha="center",
                va="bottom",
                fontsize=9
            )
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(
            save_path,
            dpi=sm.dpi,
            transparent=sm.transparent,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    
    return fig


def create_single_violin_plot(
    df: pd.DataFrame,
    column: str,
    diagnosis_col: str = 'diagnosi_definita',
    colors_dict: dict = None,
    figsize: tuple = (8, 6),
    title: str = None,
    ylabel: str = None
) -> plt.Figure:
    """
    Create a single violin plot for a specified column by diagnosis.
    
    Shows distribution shape (violin), quartiles/median (box), and 
    individual data points (scatter) for one variable across diagnoses.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    column : str
        Name of the column to visualize.
    diagnosis_col : str, default='diagnosi_definita'
        Column name containing diagnosis labels.
    colors_dict : dict, optional
        Dictionary mapping diagnosis labels to colors.
        If None, uses default Okabe-Ito palette.
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generates from column name.
    ylabel : str, optional
        Y-axis label. If None, uses column name.
    
    Returns
    -------
    fig : plt.Figure
        The created figure object.
    
    Examples
    --------
    >>> fig = create_single_violin_plot(
    ...     df, 
    ...     'eta_attuale',
    ...     colors_dict={'MSA-P': '#0072B2', 'MSA-C': '#D55E00', 'PD': '#009E73'}
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    diagnoses_order = ['MSA-P', 'MSA-C', 'PD']
    
    # Use default colors if not provided
    if colors_dict is None:
        colors_dict = {
            'MSA-P': sm.palette[0],
            'MSA-C': sm.palette[1],
            'PD': sm.palette[2]
        }
    
    # Create violin plot
    parts = ax.violinplot(
        [df[df[diagnosis_col] == diag][column].dropna() 
         for diag in diagnoses_order],
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
        [df[df[diagnosis_col] == diag][column].dropna() 
         for diag in diagnoses_order],
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
        data = df[df[diagnosis_col] == diag][column].dropna()
        y = data.values
        x = np.random.normal(i, 0.04, size=len(y))  # Add jitter
        ax.scatter(x, y, alpha=0.6, s=40, color=colors_dict[diag], 
                  edgecolors='black', linewidth=0.5, zorder=3)
    
    # Customize
    ax.set_xticks(range(len(diagnoses_order)))
    ax.set_xticklabels(diagnoses_order, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel else column, fontsize=12, 
                  fontweight='bold')
    ax.set_title(
        title if title else f'{column} distribution by diagnosis',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add mean ± SD annotations
    for i, diag in enumerate(diagnoses_order):
        data = df[df[diagnosis_col] == diag][column].dropna()
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


def create_single_histogram_plot(
    df: pd.DataFrame,
    column: str,
    diagnosis_col: str = 'diagnosi_definita',
    colors_dict: dict = None,
    figsize: tuple = (8, 6),
    bins: int = 10,
    title: str = None,
    xlabel: str = None
) -> plt.Figure:
    """
    Create a single histogram with KDE overlay for a specified column.
    
    Shows the distribution shape clearly with histogram bars and 
    kernel density estimation curves for each diagnosis group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    column : str
        Name of the column to visualize.
    diagnosis_col : str, default='diagnosi_definita'
        Column name containing diagnosis labels.
    colors_dict : dict, optional
        Dictionary mapping diagnosis labels to colors.
        If None, uses default Okabe-Ito palette.
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches.
    bins : int, default=10
        Number of histogram bins.
    title : str, optional
        Plot title. If None, auto-generates from column name.
    xlabel : str, optional
        X-axis label. If None, uses column name.
    
    Returns
    -------
    fig : plt.Figure
        The created figure object.
    
    Examples
    --------
    >>> fig = create_single_histogram_plot(
    ...     df, 
    ...     'durata_malattia',
    ...     bins=15,
    ...     colors_dict={'MSA-P': '#0072B2', 'MSA-C': '#D55E00', 'PD': '#009E73'}
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    diagnoses_order = ['MSA-P', 'MSA-C', 'PD']
    
    # Use default colors if not provided
    if colors_dict is None:
        colors_dict = {
            'MSA-P': sm.palette[0],
            'MSA-C': sm.palette[1],
            'PD': sm.palette[2]
        }
    
    # Create histograms with KDE overlay
    for diag in diagnoses_order:
        data = df[df[diagnosis_col] == diag][column].dropna()
        ax.hist(data, bins=bins, alpha=0.4, label=f'{diag} (n={len(data)})',
               color=colors_dict[diag], edgecolor='black', linewidth=1.5)
        
        # Add KDE
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = kde(x_range) * len(data) * (
                data.max() - data.min()
            ) / bins
            ax.plot(x_range, kde_values, color=colors_dict[diag], 
                   linewidth=2.5)
    
    ax.set_xlabel(xlabel if xlabel else column, fontsize=12, 
                  fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(
        title if title else f'{column} distribution by diagnosis',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig