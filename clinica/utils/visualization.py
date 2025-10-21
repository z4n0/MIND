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
