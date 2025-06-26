def extract_labels_meaning(labels, class1_name, class0_name):
  """
  Convert binary labels to meaningful class names.

  This function takes a list of binary labels (0s and 1s) and maps them to their corresponding class names.

  Parameters:
  ----------
  labels : list of int
    Binary labels (0s and 1s) to be converted
  class1_name : str
    Name/label to assign to values of 1
  class0_name : str 
    Name/label to assign to values of 0

  Returns:
  -------
  list of str
    List of class names corresponding to the input labels

  Examples:
  --------
  labels = [1, 0, 1, 0]
  extract_labels_meaning(labels, 'MSA', 'PD')
  ['MSA', 'PD', 'MSA', 'PD']
  """
  # take a list of INTEGERS NOT STRINGS and returns a list of strings containing the associated class names
  return [class1_name if label == 1 else class0_name for label in labels]

import os
import re
from cv2 import filterHomographyDecompByVisibleRefpoints
import numpy as np
def assign_class_from_image_paths(image_paths):
    """
    For each image path in image_paths, extract an integer ID from the basename
    and assign a class label based on which list (msa_lif_ids, ctrl_lif_ids, msac_ids, msap_ids, pd_ids)
    contains that ID.
    
    Args:
        image_paths (list): List of file paths to images.
        msa_lif_ids (list or set): IDs associated with the MSA LIF class.
        ctrl_lif_ids (list or set): IDs associated with the CTRL LIF class.
        msac_ids (list or set): IDs associated with the MSAC class.
        msap_ids (list or set): IDs associated with the MSAP class.
        pd_ids (list or set): IDs associated with the PD class.
    
    Returns:
        list: A list of class label strings corresponding to each image.
    """
    msa_ids = [4092, 4121, 5349, 5358, 5435, 5463, 5717, 5745, 5753, 5767, 5776, 5878, 
                5881, 5904, 5954, 5969, 5978, 5992, 5996, 6046, 6050, 6053, 6060, 6085, 6179]
    ctrl_ids = [4115, 5167, 5168, 5197, 5199, 5888, 6731, 6743, 7141]
    msac_ids    = [7239, 7293]  # example list (expand as needed)
    msap_ids    = [4121, 5349, 5358, 5435, 5717, 5745, 5753, 5767, 5776, 5878, 5978, 5992, 6050, 6053, 6179, 7144, 7120]
    pd_ids      = [6008, 6320, 6323, 6337, 6340, 6351, 6366, 6459, 6577, 6616, 6690, 6696, 6773, 7155, 7222, 7229, 7284]

    class_labels = []
    for path in image_paths:
        basename = os.path.basename(path)
        # Extract the first group of digits found in the filename
        match = re.search(r'(\d+)', basename)
        if match:
            id_num = int(match.group(1))
        else:
            id_num = None
        
        # Determine the class based on which list contains the ID
        if id_num is not None:
            if id_num in msap_ids:
                label = "MSAP"
            # elif id_num in ctrl_ids:
            #     label = "CTRL"
            # elif id_num in msac_ids:
            #     label = "MSAC"
            # elif id_num in msa_ids:
            #     label = "MSA"
            elif id_num in pd_ids:
                label = "PD"
            else:
                label = "Unknown"
        else:
            label = "Unknown"
        class_labels.append(label)
    return np.array(class_labels)
  
def remove_non_gland_images(image_paths):
  """
  Filter and return only image paths associated with glandular tissue.

  This function examines the provided list of image paths and retains only those whose filenames contain 'gh',
  indicating glandular tissue. It also prints the number of glandular images before and after filtering.

  Args:
    image_paths (list): List of file paths to images.

  Returns:
    list: Filtered list of image paths associated with glandular tissue.

  Notes:
    - The function assumes that glandular images have 'gh' in their filename.
    - Prints the count of glandular images before and after filtering.
  """
  only_gland_paths = [image for image in image_paths if "gh" in os.path.basename(image)]
  # Filter out images that contain 'vaso' (if needed)
  image_without_vaso_paths = [path for path in image_paths if 'vaso' not in os.path.basename(path).lower()]
  assert len(image_without_vaso_paths) == len(only_gland_paths), "there must be other things other than vaso and glands"
  gh_count_after = sum('gh' in os.path.basename(path).lower() for path in image_without_vaso_paths)
  # vaso_count_after = sum('vaso' in os.path.basename(path).lower() for path in image_without_vaso_paths)
  print(f"Number of glandular images before filtering: {len(image_paths)}")
  print(f"Number of glandular images after filtering: {gh_count_after}")
  return only_gland_paths
    
