# Add this code to your notebook cell

# Helper functions for architecture-based row coloring
def _get_architecture_color(model_name: str) -> str:
    """
    Map model name to LaTeX color based on architecture.
    
    Returns a LaTeX color definition compatible with xcolor package.
    Colors follow the thesis plotting style architecture_colors.
    """
    if not model_name:
        return None
    
    model_lower = str(model_name).lower()
    
    # Architecture color mapping from plotting rules
    if 'vit' in model_lower:
        return 'orange!20'  # #E69F00 -> orange!20 (light amber)
    elif 'resnet18' in model_lower:
        return 'gray!15'    # #999999 -> gray!15 (light gray)
    elif 'densenet121' in model_lower:
        return 'brown!20'   # #8B4513 -> brown!20 (light saddle brown)
    elif 'densenet169' in model_lower:
        return 'blue!15'    # #6A5ACD -> blue!15 (light slate blue)
    else:
        return None  # No color for unknown architectures


def _add_row_colors(table_tex: str, model_names: List[str]) -> str:
    """
    Add \\rowcolor commands to LaTeX table based on model names.
    
    Args:
        table_tex: LaTeX table string from pandas to_latex()
        model_names: List of model names in the same order as table rows
        
    Returns:
        Modified LaTeX string with row colors
    """
    lines = table_tex.split('\n')
    result = []
    row_idx = 0  # Track which data row we're on
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a data row (starts with content, ends with \\)
        # Data rows come after header rows and don't start with \hline or \end
        is_data_row = (line.strip() and 
                       not line.strip().startswith('\\') and 
                       '&' in line and 
                       line.strip().endswith('\\\\'))
        
        if is_data_row:
            # Get the color for this row
            if row_idx < len(model_names):
                color = _get_architecture_color(model_names[row_idx])
                if color:
                    result.append(f'\\rowcolor{{{color}}}')
                row_idx += 1
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


# Modify the write_tex_pdf_png function:
# 1. Extract model names before sorting:
def write_tex_pdf_png(key: str, df: pd.DataFrame) -> None:
    cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    if not cols:
        print(f"⚠ No matching columns in {key}; skipping.")
        return

    sub = df.loc[:, cols].copy()
    
    # Extract model names before sorting/renaming (for row coloring)
    model_names_for_colors = None
    if "model_name" in sub.columns:
        model_names_for_colors = sub["model_name"].tolist()
        sub = sub.sort_values(by="model_name")
        # Update model names list to match sorted order
        model_names_for_colors = sub["model_name"].tolist()

    sub.columns = [_pretty_header(c) for c in sub.columns]

    # ... rest of the function stays the same until after _tint_header_if_possible ...
    
    # Then add after _tint_header_if_possible:
    # Add architecture-based row colors if model_name column exists
    if model_names_for_colors:
        table_tex = _add_row_colors(table_tex, model_names_for_colors)

