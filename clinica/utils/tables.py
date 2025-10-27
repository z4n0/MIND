"""
Table generation utilities for clinical data analysis.

Provides functions to create publication-ready tables as images.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def create_table_image(
    df_table,
    title,
    filename,
    auto_width=True,
    max_col_width=0.25,
    min_col_width=0.08,
    wrap_text=True,
    max_chars_per_line=30
):
    """
    Create a publication-ready table as an image with automatic overflow
    handling.
    
    Parameters
    ----------
    df_table : pd.DataFrame
        The dataframe to visualize
    title : str
        Title for the table
    filename : str
        Output filename
    auto_width : bool, default=True
        Automatically adjust column widths based on content
    max_col_width : float, default=0.25
        Maximum column width (relative to figure width)
    min_col_width : float, default=0.08
        Minimum column width (relative to figure width)
    wrap_text : bool, default=True
        Enable text wrapping for long content
    max_chars_per_line : int, default=30
        Maximum characters per line before wrapping
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    
    Notes
    -----
    The table is saved in the images/tables folder as filename.png.
    """
    import textwrap
    
    # Define the save path
    save_path = '../images/tables/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    filename = os.path.join(save_path, filename)

    # Calculate optimal column widths
    def get_text_length(text):
        """Get effective length of text content."""
        return len(str(text))
    
    if auto_width:
        col_widths = []
        for col in df_table.columns:
            # Get max length in column (header + all values)
            header_len = get_text_length(col)
            max_val_len = df_table[col].astype(str).apply(
                get_text_length
            ).max()
            max_len = max(header_len, max_val_len)
            
            # Convert to relative width (heuristic: 1 char ≈ 0.012 width)
            width = min(max(max_len * 0.012, min_col_width), max_col_width)
            col_widths.append(width)
    else:
        # Equal widths
        col_widths = [0.15] * len(df_table.columns)
    
    # Calculate figure size dynamically
    total_width = sum(col_widths) * 100  # Convert to inches (approx)
    fig_width = max(12, min(total_width, 20))  # Between 12 and 20 inches
    fig_height = max(
        len(df_table) * 0.5 + 3,  # Changed from +2 to +3 for extra space for title
        6
    )  # Min 6 inches height
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data with intelligent text wrapping
    def wrap_cell_text(text, col_width, is_header=False):
        """
        Wrap text to fit in cell based on actual column width.
        
        Parameters
        ----------
        text : str
            Text to wrap
        col_width : float
            Column width (relative to figure)
        is_header : bool
            Whether this is a header cell
        """
        if not wrap_text:
            return str(text)
        
        text_str = str(text)
        
        # Estimate characters that fit based on column width
        # Heuristic: col_width of 0.012 ≈ 1 character
        # So reverse: chars_that_fit ≈ col_width / 0.012
        chars_per_line = int(col_width / 0.011)  # Slightly conservative
        
        # For headers, be more conservative to avoid unnecessary wrapping
        if is_header:
            chars_per_line = int(chars_per_line * 0.9)
        
        # Only wrap if text is actually longer than what fits
        if len(text_str) <= chars_per_line:
            return text_str
        
        # Wrap with intelligent line breaks
        return '\n'.join(textwrap.wrap(
            text_str,
            width=max(chars_per_line, 15),  # Min 15 chars per line
            break_long_words=False,
            break_on_hyphens=False
        ))
    
    table_data = []
    # Wrap headers based on their column widths
    table_data.append([
        wrap_cell_text(col, col_widths[i], is_header=True)
        for i, col in enumerate(df_table.columns)
    ])
    # Wrap data rows based on their column widths
    for idx, row in df_table.iterrows():
        wrapped_row = [
            wrap_cell_text(val, col_widths[i], is_header=False)
            for i, val in enumerate(row)
        ]
        table_data.append(wrapped_row)
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',  # Center all cell content horizontally
        loc='center',
        colWidths=col_widths
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.0)  # Increased from 2.2 to 3.0 for more vertical spacing
    
    # Header row styling
    for i in range(len(df_table.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')  # Professional blue
        cell.set_text_props(
            weight='bold',
            color='white',
            fontsize=11,
            ha='center',
            va='center'
        )
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
        # Add padding to header
        cell.PAD = 0.08  # Horizontal padding
    
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
            cell.set_text_props(fontsize=10, ha='center', va='center')
            # Add padding to all cells
            cell.PAD = 0.08  # Horizontal padding (default is 0.0)
    
    # Add title with more padding
    plt.title(title, fontsize=14, fontweight='bold', pad=30)  # Increased from 20 to 30
    
    # Adjust layout to prevent title overlap
    plt.subplots_adjust(top=0.95)  # Add this line to make room for title
    
    # Save with high quality
    try:
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.5,  # Increased from 0.3 to 0.5 for more padding
            facecolor='white',
            edgecolor='none'
        )
    except Exception as e:
        print(f"❌ Error saving table image: {e}")

    print(f"✅ Table saved: {filename}")
    plt.show()
    plt.close(fig)
    
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
