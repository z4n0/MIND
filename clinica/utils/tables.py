"""
Table generation utilities for clinical data analysis.

Provides functions to create publication-ready tables as images.
"""

import pandas as pd
import matplotlib.pyplot as plt

def create_table_image(df_table, title, filename):
    """
    Create a publication-ready table as an image.
    
    Parameters
    ----------
    df_table : pd.DataFrame The dataframe to visualize
    title : str Title for the table
    filename : str Output filename
    """
    fig, ax = plt.subplots(figsize=(12, len(df_table) * 0.5 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table_data = []
    table_data.append(list(df_table.columns))  # Header
    for idx, row in df_table.iterrows():
        table_data.append(list(row))
    
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(df_table.columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Header row styling
    for i in range(len(df_table.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')  # Professional blue
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Data rows styling - alternating colors
    for i in range(1, len(table_data)):
        for j in range(len(df_table.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')  # Light gray
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.5)
            cell.set_text_props(fontsize=11)
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save with high quality
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )
    print(f"✅ Table saved: {filename}")
    plt.show()
    
    return fig

# =============================================================================
# FIGURE 3: STATISTICAL SUMMARY TABLE
# =============================================================================
def create_summary_table(df, age_col, duration_col, diagnosis_col):
    """
    Create a comprehensive statistical summary table.
    """
    summary_data = []
    diagnoses_order = ['MSA-P', 'MSA-C', 'PD']
    min_plausible_age = 25
    
    for diag in diagnoses_order:
        #take rows with such diagnosis diag
        diag_df = df[df[diagnosis_col] == diag]
        
        # Age statistics returns a series of age values
        age_data = diag_df[age_col].dropna()
        # Drop rows where age is less than
        age_data = age_data[age_data >= min_plausible_age]
        age_statistics = {
            'mean': age_data.mean(),
            'std': age_data.std(),
            'median': age_data.median(),
            'min': age_data.min(),
            'max': age_data.max(),
            'n': len(age_data)
        }
        
        # Duration statistics
        dur_data = diag_df[duration_col].dropna()
        dur_statistics = {
            'mean': dur_data.mean(),
            'std': dur_data.std(),
            'median': dur_data.median(),
            'min': dur_data.min(),
            'max': dur_data.max(),
            'n': len(dur_data)
        }
        
        
        summary_data.append({
            'Diagnosis': diag,
            'N_patients': age_statistics['n'],
            'Age Mean±SD': f'{age_statistics["mean"]:.1f}±{age_statistics["std"]:.1f}',
            'Age Median': f'{age_statistics["median"]:.1f}',
            'Age Range': f'{age_statistics["min"]:.0f}-{age_statistics["max"]:.0f}',
            'N_patients': dur_statistics['n'],
            'Duration Mean±SD': f'{dur_statistics["mean"]:.1f}±{dur_statistics["std"]:.1f}',
            'Duration Median': f'{dur_statistics["median"]:.1f}',
            'Duration Range': f'{dur_statistics["min"]:.0f}-{dur_statistics["max"]:.0f}'
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df
