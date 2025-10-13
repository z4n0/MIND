import re
import numpy as np
import os

## TODO this whole file could potentially be deleted


def filter_paths_by_imageIds(image_paths, labels, excluded_ids):
    """
    Filters out images that contain specified index numbers in their paths.
    
    Args:
        image_paths (np.ndarray): Array of image paths
        labels (np.ndarray): Array of corresponding labels
        excluded_indexes (list): List of index numbers to exclude (e.g., [1,2,3])
        
    Returns:
        tuple: (filtered_paths, filtered_labels)
    """
    # Convert excluded_indexes to strings for comparison
    excluded = [str(idx) for idx in excluded_ids]
    
    # Debug: Print a few paths to understand the format
    print("Sample paths:")
    for path in image_paths[:2]:
        print(path)
    
    # Create mask for filtering - more flexible pattern matching
    mask = []
    for path in image_paths:
        # Extract all numbers from the path
        path_numbers = re.findall(r'(\d+)', str(path))
        # Check if any excluded index is in the path numbers
        should_keep = not any(idx in path_numbers for idx in excluded)
        mask.append(should_keep)
    
    mask = np.array(mask)
    
    # Debug: Print matching information
    print("\nDebug info:")
    print(f"Total paths: {len(image_paths)}")
    print(f"Excluded indexes: {excluded_ids}")
    print(f"Matches found: {len(image_paths) - sum(mask)}")
    
    # Apply mask to both paths and labels
    filtered_paths = image_paths[mask]
    filtered_labels = labels[mask]
    
    # Print summary
    removed = len(image_paths) - len(filtered_paths)
    print(f"\nSummary:")
    print(f"Removed {removed} images containing indexes: {excluded_ids}")
    print(f"Remaining images: {len(filtered_paths)}")
    
    return filtered_paths, filtered_labels

def filter_paths_by_classIndex(images_paths_np, labels_np, indexToRemove):
    """
    Filter out image paths based on their class index.
    
    Args:
        images_paths_np (np.ndarray): Array containing the image file paths.
        labels_np (np.ndarray): Array containing corresponding labels (0 or 1).
        classIndex (int): Index of the class to keep.
    
    Returns:
        np.ndarray: Filtered array of image paths.
        np.ndarray: Filtered array of labels.
    """
    # Convert inputs to lists      
    mask = (labels_np == indexToRemove) #returns a boolean array with True where the label is 1
    filtered_paths = images_paths_np[mask] #boolean filtering
    filtered_labels = labels_np[mask]
    return filtered_paths, filtered_labels

def get_label(path, class1_ids, class0_ids):
    """
    Determine the label of a file based on its filename.
    This function extracts 4-digit IDs from the filename and checks them against
    provided lists of IDs to determine the label.
    Parameters:
    path (str): The file path from which to extract the filename.
    class1_ids (set or list): A collection of IDs that should be labeled as 1.
    class0_ids (set or list): A collection of IDs that should be labeled as 0.
    Returns:
    int: The label of the file (1 or 0).
    Raises:
    ValueError: If no 4-digit ID is found in the filename or if the ID is not in
                either class1_ids or class0_ids.
    """
    
    filename = os.path.basename(path)
    
    # Extract only 4-digit numbers from the filename
    id_matches = re.findall(r'\d{4}', filename)
    
    if not id_matches:
        raise ValueError(f"No 4-digit ID found in filename: {filename}")
    
    # Convert to integers and check against test sets
    for id_str in id_matches:
        id_num = int(id_str)
        if id_num in class1_ids:
            return 1
        elif id_num in class0_ids:
            return 0
    
    # If no match was found
    raise ValueError(f"Unknown 4-digit ID in filename: {filename}")