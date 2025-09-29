import numpy as np
import matplotlib.pyplot as plt 
from utils.data_extraction_functions import extract_labels_meaning
import os, re, torch, tifffile, random


def get_image_array(item):
    """
    Given an item (either a file path, numpy array, or torch.Tensor),
    return a numpy array of type float32 representing the image.
    """
    if isinstance(item, str):
        try:
            # Read the image from file
            image = tifffile.imread(item)
        except Exception as e:
            print(f"Error reading image at {item}: {e}")
            return None
        return image.astype(np.float32)
    elif isinstance(item, torch.Tensor):
        return item.cpu().numpy().astype(np.float32)
    elif isinstance(item, np.ndarray):
        return item.astype(np.float32)
    else:
        raise ValueError("Unsupported type in images_or_tensors")
    
def visualize_and_compare_pixel_intensity_histograms(
    images_or_tensors: list,
    labels: list,
    class1_name: str,
    class0_name: str,
    bins: int = 256
) -> None:
    """
    Visualizes and compares the average pixel intensity histograms for two classes of images.

    This function computes per-channel histograms for each image, averages them by class,
    normalizes them to relative frequencies, and plots the results for visual comparison.

    Args:
        images_or_tensors (list): List of image file paths, numpy arrays.
        labels (list): List of integer labels (0 or 1) corresponding to each image.
        class1_name (str): Name of class 1 (label 1).
        class0_name (str): Name of class 0 (label 0).
        bins (int, optional): Number of bins for the histograms. Default is 256.

    Raises:
        ValueError: If no images are found for class 0 or class 1.

    Returns:
        None. Displays the histogram plots using matplotlib.
    """
    histograms_class_0 = []
    histograms_class_1 = []
    all_images = []  # To store the actual image arrays for global min/max

    for item, label in zip(images_or_tensors, labels):
        image_array = get_image_array(item)
        if image_array is None:
            continue
        # Save the image for global min/max calculation
        all_images.append(image_array)

        # Convert to torch.Tensor if needed for your histogram function
        image_tensor = torch.from_numpy(image_array)
        histograms = calculate_tensor_histogram(image_tensor, bins)
        if histograms is None:
            continue
        if label == 0:
            histograms_class_0.append(histograms)
        else:
            histograms_class_1.append(histograms)

    # Convert lists to arrays and average
    histograms_class_0 = np.array(histograms_class_0)
    histograms_class_1 = np.array(histograms_class_1)

    if histograms_class_0.size > 0:
        avg_histograms_class_0 = np.mean(histograms_class_0, axis=0)
    else:
        raise ValueError("No images found for class 0")

    if histograms_class_1.size > 0:
        avg_histograms_class_1 = np.mean(histograms_class_1, axis=0)
    else:
        raise ValueError("No images found for class 1")

    # Normalize to relative frequencies
    avg_histograms_class_0 = avg_histograms_class_0 / np.sum(avg_histograms_class_0, axis=1, keepdims=True)
    avg_histograms_class_1 = avg_histograms_class_1 / np.sum(avg_histograms_class_1, axis=1, keepdims=True)

    # Compute the overall global min and max from the actual image arrays
    global_min = min(img.min() for img in all_images)
    global_max = max(img.max() for img in all_images)
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the averaged histograms using the computed bin centers
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    channel_names = ['R', 'G', 'B']

    # Example: Assuming extract_labels_meaning returns proper labels.
    label_0 = extract_labels_meaning([0], class1_name, class0_name)[0]
    label_1 = extract_labels_meaning([1], class1_name, class0_name)[0]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.bar(
            bin_centers,
            avg_histograms_class_0[i],
            width=bin_edges[1] - bin_edges[0],
            label=label_0,
            color='blue',
            alpha=0.7
        )
        plt.bar(
            bin_centers,
            avg_histograms_class_1[i],
            width=bin_edges[1] - bin_edges[0],
            label=label_1,
            color='red',
            alpha=0.5
        )
        plt.title(f"{channel_names[i]} Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Relative Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()


def print_class_statistics(class_paths, class_name):
    """
    Print statistics about image types within a class.

    Args:
        class_paths (list): List of image paths for a specific class
        class_name (str): Name of the class
    """
    gh_count = sum('gh' in path.lower() for path in class_paths)
    vaso_count = sum('vaso' in path.lower() for path in class_paths)
    total_count = len(class_paths)

    print(f"\nClass: {class_name}")
    print(f"Total images: {total_count}")
    print(f"'gh' images: {gh_count} ({(gh_count/total_count)*100:.1f}%)")
    print(f"'vaso' images: {vaso_count} ({(vaso_count/total_count)*100:.1f}%)")
    print("-" * 40)


def print_image_statistic(image_paths, image_labels, num_samples, class1_name, class0_name):
    """
    Check if images are normalized and print their labels.

    Args:
        image_paths (list): List of paths to TIFF images
        image_labels (list): List of corresponding labels
        num_samples (int): Number of random images to check

    Returns:
        None (prints normalization statistics)
    """
    # Sample random indices
    indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    check_paths = [image_paths[i] for i in indices]
    check_labels = [image_labels[i] for i in indices]

    print(f"Checking {len(check_paths)} images for normalization:\n")

    for idx, (path, label) in enumerate(zip(check_paths, check_labels), 1):
        # Load image
        img = tifffile.imread(path)
        print("Image shape: ", img.shape)

        # Global statistics
        min_val = np.min(img)
        max_val = np.max(img)
        mean_val = np.mean(img)
        std_val = np.std(img)

        # Determine if likely normalized globally
        is_zero_one = (min_val >= -0.1 and max_val <= 1.1)
        is_standardized = (abs(mean_val) < 5 and abs(std_val - 1) < 2)
        label =  extract_labels_meaning([label],class1_name,class0_name)[0]
        print(f"\nImage {idx}: {path.split('/')[-1]}")
        print(f"Label: {label}")
        print("\nGlobal Statistics:")
        print(f"  Min: {min_val:.3f}")
        print(f"  Max: {max_val:.3f}")
        print(f"  Mean: {mean_val:.3f}")
        print(f"  Std: {std_val:.3f}")
        print(f"  Likely [0,1] normalized: {is_zero_one}")
        print(f"  Likely standardized: {is_standardized}")

        # Channel-wise statistics
        print("\nChannel-wise Statistics:")
        for c in range(img.shape[0]):
            channel = img[c]
            ch_min = np.min(channel)
            ch_max = np.max(channel)
            ch_mean = np.mean(channel)
            ch_std = np.std(channel)

            ch_is_zero_one = (ch_min >= -0.1 and ch_max <= 1.1)
            ch_is_standardized = (abs(ch_mean) < 5 and abs(ch_std - 1) < 2)

            print(f"\nChannel {c}:")
            print(f"  Min: {ch_min:.3f}")
            print(f"  Max: {ch_max:.3f}")
            print(f"  Mean: {ch_mean:.3f}")
            print(f"  Std: {ch_std:.3f}")
            print(f"  Likely [0,1] normalized: {ch_is_zero_one}")
            print(f"  Likely standardized: {ch_is_standardized}")
        print("\n" + "-"*50)

def visualize_tiff(image_path, channel_wise_norm=True, figsize=(10,10)):
    """
    Visualize a TIFF image with proper scaling and channel ordering.
    Applies channel-wise min-max scaling for visualization by default
    else if channel_wise_norm is False it does global min-max scaling.
    
    Handles both 3-channel (Green, Blue, Red) and 4-channel (Green, Blue, Grey, Red) images.

    Args:
        image_path (str): Path to TIFF image
        channel_wise_norm (bool): Whether to apply channel-wise normalization
        figsize (tuple): Figure size for display
    """
    # Read the image
    img = tifffile.imread(image_path)

    # Transpose from (C,H,W) to (H,W,C) for matplotlib
    if img.shape[0] <= 4:
        img = np.transpose(img, (1,2,0))

    # Use the min_max_normalization function with channel_wise=True
    scaled_img = min_max_normalization(img, channel_wise=channel_wise_norm)

    # Determine channel names based on channel count
    num_channels = img.shape[-1]
    if num_channels == 3:
        channel_names = ['Green', 'Blue', 'Red']
        # Reorder channels from [Green, Blue, Red] to [Red, Green, Blue] for display
        display_img = scaled_img[:,:,[2,0,1]]
    elif num_channels == 4:
        channel_names = ['Green', 'Blue', 'Grey', 'Red']
        # Reorder channels from [Green, Blue, Grey, Red] to [Red, Green, Blue] for display
        # We'll drop the Grey channel as RGB image can only show 3 channels
        display_img = scaled_img[:,:,[3,0,1]]
    else:
        channel_names = [f'Channel {i}' for i in range(num_channels)]
        display_img = scaled_img  # No reordering for generic case

    plt.figure(figsize=figsize)
    if num_channels <= 3:
        plt.imshow(display_img)
    elif num_channels == 4:
        plt.imshow(display_img)
        print("Note: Grey channel is not shown in the RGB visualization")
    else:
        plt.imshow(display_img[:,:,:3])  # Show only first 3 channels
        print(f"Note: Only showing 3 out of {num_channels} channels")
    plt.axis('off')
    plt.title('Combined RGB Image')
    plt.show()

    # Print value ranges for reference
    print("\nOriginal value ranges:")
    
    for i, name in enumerate(channel_names):
        if i < num_channels:  # Only print for channels that exist
            channel = img[..., i]
            print(f"{name}: min={np.min(channel):.2f}, max={np.max(channel):.2f}")


# Now do a min-max scale for display, so negative => 0, max => 1
def min_max_normalization(array, channel_wise=False):
    """
    Normalizes an array to the range [0,1] using min-max scaling.
    Can perform either global or channel-wise normalization.
    
    Args:
        array (numpy.ndarray): Input array to normalize.
        channel_wise (bool): If True, normalize each channel independently.
                              If False, normalize using global min/max.
    
    Returns:
        numpy.ndarray: Normalized array in range [0,1].
                       Returns zeros if the range is too small.
    """
    if not channel_wise:
        # Global normalization (original behavior)
        a_min, a_max = array.min(), array.max()
        rng = a_max - a_min
        if rng < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return (array - a_min) / rng
    
    # Channel-wise normalization
    normalized = np.zeros_like(array, dtype=np.float32)
    
    # Ensure the input has 3 dimensions (e.g., (C, H, W) or (H, W, C)).
    if array.ndim != 3:
        raise ValueError("Channel-wise normalization requires 3D input")
        
    # Determine the channel axis:
    # If the first dimension is 3 or 4, assume channel-first.
    # Otherwise, if the last dimension is 3 or 4, assume channel-last.
    if array.shape[0] in (3, 4):
        channel_axis = 0
    elif array.shape[-1] in (3, 4):
        channel_axis = -1
    else:
        raise ValueError("Input image must have 3 or 4 channels in either the first or last axis")
    
    n_channels = array.shape[channel_axis]
    
    # Normalize each channel independently.
    for c in range(n_channels):
        if channel_axis == 0:
            channel = array[c, ...]
            c_min, c_max = channel.min(), channel.max()
            rng = c_max - c_min
            if rng < 1e-8:
                normalized[c, ...] = 0
            else:
                normalized[c, ...] = (channel - c_min) / rng
        else:
            channel = array[..., c]
            c_min, c_max = channel.min(), channel.max()
            rng = c_max - c_min
            if rng < 1e-8:
                normalized[..., c] = 0
            else:
                normalized[..., c] = (channel - c_min) / rng
    
    return normalized.astype(np.float32)


def imageNet_denormalize(img):
    """
    Denormalizes an image that was normalized using ImageNet statistics.
    
    Args:
        img (numpy.ndarray): Image in format [H, W, C] or [C, H, W] with values normalized 
                          using ImageNet mean and std
    
    Returns:
        numpy.ndarray: Denormalized image with pixel values in range [0, 1]
    """
    # Standard ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Check if image is in [C, H, W] format and convert to [H, W, C] for processing
    channels_first = (img.shape[0] == 3)
    if channels_first:
        img = img.transpose(1, 2, 0)
    
    # Denormalize: multiply by std and add mean
    if img.shape[-1] == 3:  # Make sure we're dealing with an RGB image
        # Reshape mean and std for broadcasting
        mean = mean.reshape(1, 1, 3)
        std = std.reshape(1, 1, 3)
        
    # Apply denormalization
    img = img * std + mean
    
    # Clip values to valid range [0, 1]
    img = np.clip(img, 0, 1)
    
    # Return in the original format
    if channels_first:
        img = img.transpose(2, 0, 1)
        
    return img

def calculate_tensor_histogram(image_tensor, bins=256, plot=False, density=True):
    """
    Calculates the histogram for each channel of an image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor
        bins (int): Number of bins for the histogram (default: 256)
        plot (bool): Whether to plot the histograms (default: False)
        density (bool): Whether to normalize the histogram (default: True)

    Returns:
        numpy.ndarray: Array of shape (3, bins) containing histograms for each channel
    """
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.cpu().numpy()
    else: # Assume it's already a NumPy array
        image_np = image_tensor
    # Add at the start of the function
    #print(f"Value ranges in input data:")
    # Transpose to channel-last if it's channel-first
    if image_np.shape[0] == 3 and image_np.ndim == 3:
        image_np = image_np.transpose(1, 2, 0)

    histograms = np.zeros((3, bins))
    channel_names = ['G', 'B', 'R']
    colors = ['green', 'blue', 'red']

    global_min = np.min(image_np)
    global_max = np.max(image_np)

    if plot:
        plt.figure(figsize=(15, 5))
        #print(f"Global min: {global_min}, Global max: {global_max}")

    for channel in range(3):
        channel_data = image_np[..., channel].flatten()
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        if min_val == max_val:
            hist = np.zeros(bins)
            hist[bins//2] = len(channel_data)
            bin_edges = np.linspace(global_min, global_max, bins + 1)
        else:
            hist, bin_edges = np.histogram(channel_data,
                                         bins=bins,
                                         range=(global_min, global_max),
                                         density=density)
        histograms[channel] = hist

        if plot:
            plt.subplot(1, 3, channel + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]

            plt.bar(bin_centers,
                   hist,
                   width=bin_width,
                   color=colors[channel],
                   alpha=0.6,
                   edgecolor=colors[channel],
                   label=channel_names[channel])

            plt.title(f"{channel_names[channel]} Channel Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.xlim(global_min, global_max)
            plt.legend()

    if plot:
        plt.tight_layout()
        plt.show()

    return histograms

#calculate_color_histograms_from_paths has been deleted for calculate_tensor_histogram
    
def visualize_dict_image(data_dict, step_name="", cmap="gray"):
    """
    Visualizes an image from a MONAI data dictionary AND its RGB channel histograms.
    Handles both single and multi-channel images.

    Args:
        data_dict (dict): MONAI data dictionary containing 'image' key
        step_name (str, optional): Title for the plot. Defaults to empty string.
        cmap (str, optional): Colormap for single channel images. Defaults to "gray".
    """
    img = data_dict["image"]  # shape torch.Size([1024, 1024, 3])
    #hist = calculate_tensor_histogram(img, plot=True)
    img_np = img.cpu().numpy()

    # If it's channel-first (C,H,W), reorder to (H,W,C) for matplotlib
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
        img_np = np.transpose(img_np, (1, 2, 0))  # => (H,W,C)
    print("visualization is done via min max scaling")
    img_display = min_max_normalization(img_np)

    # print(
    #     f"{step_name} shape={img_np.shape}, raw range=({img_np.min():.3f},{img_np.max():.3f}), "
    #     f"display range=({img_display.min():.3f},{img_display.max():.3f})"
    # )
    plt.imshow(img_display)
    plt.title(step_name)
    plt.axis("off")
    plt.show()

# Add learning curves visualization
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, ignore_first_epoch_loss=False):
    """ 
    Plots learning curves for training and validation losses and accuracies.
    NOTE it could be that train/val_accuracies are balanced accuracy not accuracy
    Args:
        train_losses (list): Training loss values
        val_losses (list): Validation loss values
        train_accuracies (list): Training accuracy values
        val_accuracies (list): Validation accuracy values
        ignore_first_epoch_loss (bool): If True, the y-axis for the loss plot is scaled
                                        based on the losses from the second epoch onwards.
    """
    
    if any(loss is None for loss in [train_losses, val_losses]):
        raise ValueError("train_losses and val_losses must be provided.")
    if any(accuracy is None for accuracy in [train_accuracies, val_accuracies]):
        raise ValueError("train_accuracies and val_accuracies must be provided.")
    
    # Create epoch numbers for x-axis
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # to solve the issue of the first epoch loss being much higher than the second epoch loss
    # making the graph visualization not clear since it have a much higher scale than needed
    # with this we use the max loss from the second epoch onwards to set the y-limit
    if ignore_first_epoch_loss and len(val_losses) > 1 and len(train_losses) > 1:
        # Find the max loss from the second epoch onwards
        max_loss_after_epoch1 = max(max(train_losses[1:]), max(val_losses[1:]))
        # Set the y-limit to be slightly above this max value, with a minimum of 0
        ax1.set_ylim(0, max_loss_after_epoch1 * 1.1)

    # Accuracy curves
    ax2.plot(epochs, train_accuracies, label='Train Acc')
    ax2.plot(epochs, val_accuracies, label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

def visualize_batch_histograms(images, labels, class1_name, class0_name, class2_name=None, bins=256):
    """
    Visualizes histograms of pixel intensity distributions for a batch of images,
    including labels in the plot titles. Supports up to three classes.

    Args:
        images: A PyTorch tensor of images (B, C, H, W).
        labels: A PyTorch tensor of labels (B,).
        class1_name: Name of class 1.
        class0_name: Name of class 0.
        class2_name: Name of class 2 (optional).
        bins: The number of bins for the histogram.
    """
    images_np = images.cpu().numpy()  # Convert to numpy array
    num_images = images_np.shape[0]
    num_channels = images_np.shape[1]

    fig, axes = plt.subplots(num_images, num_channels, figsize=(5 * num_channels, 5 * num_images))

    for i in range(num_images):
        label_value = labels[i].item()
        if label_value == 0:
            label = class0_name
        elif label_value == 1:
            label = class1_name
        elif class2_name is not None and label_value == 2:
            label = class2_name
        else:
            label = f"Unknown (Label: {label_value})"

        for j in range(num_channels):
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.hist(images_np[i, j].flatten(), bins=bins, color='skyblue', alpha=0.7)
            ax.set_title(f"Image {i+1} (Label: {label}), Channel {j+1}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# --- Normalization function inspired by the reference ---
def normalize_image(image, per_channel=False):
    """
    Normalizes image channels to [0, 1]. Handles constant channels.
    Expects image to be float dtype.
    """
    if image.ndim == 3 and per_channel:
        normalized_img = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):
            ch = image[..., c]
            min_val = np.min(ch) # Use np.min/max for better NaN handling if needed
            max_val = np.max(ch)
            if max_val > min_val:
                normalized_img[..., c] = (ch - min_val) / (max_val - min_val)
            elif max_val == min_val and max_val != 0: # Keep constant non-zero value scaled to 1? Or keep as is? Let's try keeping original scaling for constants.
                # Or maybe scale constant channels to 0.5? Or 1?
                # Reference function just assigned 'ch' back. Let's try that approach.
                # If input was float, keep it. If int, it might be large.
                # Safest is scaling constant channels to 0 or 0.5? Let's scale to 0 for simplicity.
                # normalized_img[..., c] = np.zeros_like(ch, dtype=np.float32)
                # Let's follow reference more closely: if max==min, just copy.
                # The display range might need adjustment later if we don't scale to [0,1]
                 normalized_img[..., c] = ch # Keep constant value
            else: # All zeros
                normalized_img[..., c] = ch # Keep zeros
        # Clip ensures outputs are generally ok, but constant channels might be outside [0,1] if not scaled
        # Re-evaluate: let's force constants to 0 or 0.5 for [0,1] consistency
        for c in range(image.shape[-1]):
             ch = image[..., c]
             min_val = np.min(ch)
             max_val = np.max(ch)
             if max_val > min_val:
                 normalized_img[..., c] = (ch - min_val) / (max_val - min_val)
             # else: # Constant channel defaults to 0 in normalized_img initialization
                 # Re-initialize to handle constants better
        normalized_img = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):
            ch = image[..., c]
            min_val = np.min(ch)
            max_val = np.max(ch)
            if max_val > min_val:
                normalized_img[..., c] = (ch - min_val) / (max_val - min_val)
            # else: constant channels remain 0
        return normalized_img

    elif image.ndim == 2: # Grayscale or Overlay
        normalized_img = np.zeros_like(image, dtype=np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized_img = (image - min_val) / (max_val - min_val)
        # else: constant remains 0
        return normalized_img
    else: # Not 2D or 3D, return as is (maybe convert to float)
        return image.astype(np.float32, copy=False)


def display_images_by_class(image_paths, labels, class_names, ncols=5, normalize=True, overlay_alpha=0.4):
    """
    Displays images grouped by class. Handles grayscale, 3ch (G,B,R -> RGB display),
    and 4ch (G,B,Gray,R -> RGB base + Gray overlay). Uses reference overlay method.

    Args:
        image_paths (np.ndarray or list): Image file paths.
        labels (np.ndarray or list): Corresponding integer labels.
        class_names (list): Names for the classes.
        ncols (int): Columns in the grid. Default is 5.
        normalize (bool): If True, apply per-channel min-max normalization to base RGB
                          and global normalization to overlay. Default is True.
                          If False, convert to float and scale by data type max for integers.
        overlay_alpha (float): Alpha transparency for the Gray channel overlay (0 to 1).
                               Default is 0.4.
    """
    # --- Input validation and setup code remains the same ---
    num_classes = len(class_names)
    max_label = -1
    if len(labels) > 0:
       try:
           numeric_labels = [l for l in labels if isinstance(l, (int, np.integer))]
           if numeric_labels:
               max_label = np.max(numeric_labels)
           else:
                pass
       except TypeError:
            raise TypeError("Labels must be numeric (integers).")
    if not isinstance(class_names, list) or num_classes == 0:
         raise ValueError("class_names must be a non-empty list.")
    if max_label != -1 and max_label >= num_classes:
        raise ValueError(f"Maximum label found ({max_label}) is out of bounds for the provided class_names (indices 0 to {num_classes-1}).")
    if not isinstance(image_paths, (list, np.ndarray)) or not isinstance(labels, (list, np.ndarray)):
         raise TypeError("image_paths and labels should be lists or numpy arrays.")
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length.")
    image_paths = list(image_paths)
    labels = list(labels)
    paths_by_class = [[] for _ in range(num_classes)]
    for path, lab in zip(image_paths, labels):
        if isinstance(lab, (int, np.integer)) and 0 <= lab < num_classes:
            paths_by_class[lab].append(path)
        else:
            filename = os.path.basename(path) if isinstance(path, str) else "Unknown"
            print(f"Warning: Skipping image '{filename}' with invalid label '{lab}'. Expected integer labels between 0 and {num_classes-1}.")
    # --- End of validation and setup ---
    # --- Inner function to plot a grid for a single class ---
    def plot_grid(image_list, title, preferred_ncols, normalize, overlay_alpha):
        if not image_list:
            print(f"No images found for {title}.")
            return

        n_images = len(image_list)
        ncols = preferred_ncols if n_images >= preferred_ncols else n_images
        nrows = int(np.ceil(n_images / ncols))
        scale_factor = 6 / n_images if n_images < 6 else 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3 * scale_factor, nrows * 3 * scale_factor), squeeze=False)
        axs = axs.flatten()

        for i, path in enumerate(image_list):
            ax = axs[i]
            filename = os.path.basename(path) if isinstance(path, str) else "Invalid Path"
            try:
                # Read image using tifffile
                img = tifffile.imread(path)
                img_orig_dtype = img.dtype # Keep original dtype info

                # --- Channel Selection and Preparation ---
                base_rgb = None
                overlay = None
                cmap = None
                is_color = False

                # Transpose C,H,W to H,W,C if necessary
                if img.ndim == 3 and img.shape[0] in [3, 4] and img.shape[-1] not in [3, 4]:
                    img = np.transpose(img, (1, 2, 0))

                if img.ndim == 2: # Grayscale
                    base_rgb = img # Will be treated as grayscale by imshow
                    cmap = 'gray'
                    is_color = False
                elif img.ndim == 3:
                    num_channels = img.shape[-1]
                    if num_channels == 3:
                        # Input: G(0), B(1), R(2) -> Output R, G, B
                        # print(f"Info: Processing 3-channel image '{filename}' as RGB.")
                        base_rgb = img[..., [2, 0, 1]]
                        is_color = True
                    elif num_channels == 4:
                        # Input: G(0), B(1), Gray(2), R(3) -> Output R, G, B base + Gray overlay
                        # print(f"Info: Processing 4-channel image '{filename}' with Gray overlay.")
                        base_rgb = img[..., [3, 0, 1]] # R, G, B
                        overlay = img[..., 2]          # Gray
                        is_color = True
                    elif num_channels == 1: # Handle shape (H, W, 1)
                         base_rgb = img[..., 0] # Treat as grayscale
                         cmap = 'gray'
                         is_color = False
                    else: # Fallback for unexpected channels
                         print(f"Warning: Image '{filename}' has {num_channels} channels. Displaying first channel as grayscale.")
                         base_rgb = img[..., 0]
                         cmap = 'gray'
                         is_color = False
                else:
                    print(f"Warning: Skipping image '{filename}' with unsupported dimensions {img.shape}.")
                    ax.set_title(f"{filename}\nUnsupported Dim", fontsize=8)
                    ax.axis('off')
                    continue

                # --- Normalization / Scaling ---
                # Convert to float for processing
                if base_rgb is not None:
                    base_rgb_float = base_rgb.astype(np.float32)
                if overlay is not None:
                    overlay_float = overlay.astype(np.float32)

                vmin, vmax = None, None # Let imshow determine range by default unless normalized

                if normalize:
                    if base_rgb is not None:
                        base_rgb_processed = normalize_image(base_rgb_float, per_channel=is_color)
                    if overlay is not None:
                        overlay_processed = normalize_image(overlay_float, per_channel=False) # Normalize overlay globally
                    # When normalized, the range is known to be [0, 1]
                    vmin, vmax = 0.0, 1.0
                else:
                    # If not normalizing, scale integers by dtype max to approx [0, 1] range
                    # Keep floats as they are (assuming they are already sensible)
                    base_rgb_processed = base_rgb_float
                    overlay_processed = overlay_float
                    if np.issubdtype(img_orig_dtype, np.integer):
                         dtype_max = np.iinfo(img_orig_dtype).max
                         if base_rgb is not None:
                             base_rgb_processed = base_rgb_float / dtype_max
                         if overlay is not None:
                             overlay_processed = overlay_float / dtype_max
                         # Clip after scaling
                         if base_rgb is not None:
                              base_rgb_processed = np.clip(base_rgb_processed, 0.0, 1.0)
                         if overlay is not None:
                              overlay_processed = np.clip(overlay_processed, 0.0, 1.0)
                         # Range is now approx [0, 1]
                         vmin, vmax = 0.0, 1.0
                    # else: for float inputs with normalize=False, let imshow auto-scale (vmin=None, vmax=None)

                # --- Display using reference method ---
                if base_rgb is not None:
                     ax.imshow(base_rgb_processed, cmap=cmap, vmin=vmin, vmax=vmax)

                if overlay is not None:
                     ax.imshow(overlay_processed, cmap='gray', alpha=overlay_alpha, vmin=vmin, vmax=vmax) # Overlay on top

                # --- Finishing Touches ---
                ax.axis('off')
                match = re.search(r"(MAX_.*?\.lif)", filename, flags=re.IGNORECASE)
                display_name = match.group(1) if match else filename
                ax.set_title(display_name, fontsize=8)

            except FileNotFoundError:
                 print(f"Error: Image file not found '{path}'")
                 ax.set_title(f"{filename}\nNot Found", fontsize=8)
                 ax.axis('off')
            except Exception as e:
                print(f"Error processing image '{path}': {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                ax.set_title(f"{filename}\nError", fontsize=8)
                ax.axis('off')
                continue

        # Turn off any unused subplots
        for j in range(n_images, len(axs)):
            axs[j].axis('off')

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
    # --- End of inner plot_grid function ---

    # Plot grids for each class
    for i in range(num_classes):
        class_title = class_names[i]
        image_list_for_class = paths_by_class[i]
        plot_grid(image_list_for_class, class_title, ncols, normalize, overlay_alpha)



def print_batch_image_statistics(images, labels, class1_name, class0_name):
    """
    Calculates and prints statistics (min, max, mean, std) for each image
    in a batch, along with the corresponding label.

    Args:
        images: A PyTorch tensor of images (B, C, H, W).
        labels: A PyTorch tensor of labels (B,).
    """

    num_images = images.shape[0]
    num_channels = images.shape[1]

    for i in range(num_images):
        label = extract_labels_meaning([labels[i].item()],class1_name,class0_name)
        print(f"Image {i+1} (Label: {label[0]}):")  # Print label
        for j in range(num_channels):
            channel_data = images[i, j].flatten()

            min_val = torch.min(channel_data).item()
            max_val = torch.max(channel_data).item()
            mean_val = torch.mean(channel_data).item()
            std_val = torch.std(channel_data).item()

            print(f"  Channel {j+1}:")
            print(f"    Min: {min_val:.4f}")
            print(f"    Max: {max_val:.4f}")
            print(f"    Mean: {mean_val:.4f}")
            print(f"    Std: {std_val:.4f}")
        print("-" * 20)

def visualize_batch_boxplots(images, labels, class1_name, class0_name):
    """
    Visualizes box plots of pixel intensity distributions for a batch of images,
    including labels in the plot titles.

    Args:
        images: A PyTorch tensor of images (B, C, H, W).
        labels: A PyTorch tensor of labels (B,).
    """

    images_np = images.cpu().numpy()
    num_images = images_np.shape[0]
    num_channels = images_np.shape[1]

    fig, axes = plt.subplots(num_images, num_channels, figsize=(5 * num_channels, 5 * num_images))

    for i in range(num_images):
        label = extract_labels_meaning([labels[i].item()],class1_name, class0_name)
        for j in range(num_channels):
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.boxplot(images_np[i, j].flatten(), vert=True, patch_artist=True, showfliers=False)
            ax.set_title(f"Image {i+1} (Label: {label[0]}), Channel {j+1}")  # Add label to title
            ax.set_ylabel("Pixel Value")
    plt.tight_layout()
    plt.show()

def generate_cv_results_figure(fold_results, prefix='test'):
    """
    Generate a boxplot figure of cross-validation results without showing it.
    
    Args:
        fold_results (list): A list of dictionaries with fold metrics.
        prefix (str): The prefix for metric keys (e.g., 'test' or 'val').
        
    Returns:
        fig (matplotlib.figure.Figure or None): The generated figure, or None if no valid metrics found.
    """
    # Define the base metric names and construct the expected keys
    base_metrics = ['loss', 'acc', 'f1', 'balanced_acc']
    metrics = [f"{prefix}_{metric}" for metric in base_metrics]
    
    # Determine which metrics exist in the fold results
    available_metrics = [metric for metric in metrics if any(metric in fold for fold in fold_results)]
    
    if not available_metrics:
        print(f"No metrics with prefix '{prefix}_' found in fold_results.")
        print(f"Available keys: {list(fold_results[0].keys())}")
        return None
    
    # Calculate subplot grid dimensions
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    
    # Ensure axes is always iterable
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    # Create a boxplot for each available metric
    for idx, metric in enumerate(available_metrics):
        values = [fold[metric] for fold in fold_results if metric in fold]
        axes[idx].boxplot(values)
        axes[idx].set_title(f"Distribution of {metric} across folds")
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].grid(True)
    
    # Remove any unused axes
    for ax in axes[len(available_metrics):]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    return fig

import seaborn as sns

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', figsize=(8,6)):
    """
    Visualize confusion matrix without immediately displaying it.

    Args:
        cm (array-like): Confusion matrix data (2D array).
        class_names (list of str): Names of classes for x-axis and y-axis.
        title (str, optional): Title of the confusion matrix plot. Default is 'Confusion Matrix'.
        figsize (tuple, optional): Size of the plot figure. Default is (8, 6).

    Returns:
        matplotlib.figure.Figure: The figure object containing the confusion matrix plot.
    """
    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using seaborn on the provided axis
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Do not call plt.show() here so the figure remains available
    return fig


def show_misclassified_images(misclassified_images, num_images=10, figsize=(20, 20), normalize=True, class_names=None, overlay_alpha_DAPI=0.2):
    """
    Display misclassified images with optional normalization.
    
    This function supports:
      - 3-channel images in order (Green, Blue, Red) by reordering to (Red, Green, Blue)
      - 4-channel images in order (Green, Blue, Gray, Red): 
           It creates an RGB composite from channels [3, 0, 1] (i.e. Red, Green, Blue) 
           and overlays the fourth (Gray) channel as a grayscale image.
    
    Args:
        misclassified_images (list): Each dictionary should include keys "image", "pred_label", and "true_label".
                                      "image" is expected to be a numpy array with shape (C, H, W) or (H, W, C).
        num_images (int): Number of images to display.
        figsize (tuple): Figure size.
        normalize (bool): Whether to apply per-channel min-max normalization.
        class_names (list of str, optional): List of class names for mapping integer labels.
        overlay_alpha (float): Transparency for overlaying the gray channel on 4-channel images.
    """
    if not misclassified_images:
        print("No misclassified images to display.")
        return

    # Determine the number of images and grid layout.
    num_images = min(num_images, len(misclassified_images))
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    fig = plt.figure(figsize=figsize)
    print("Keys in misclassified sample:", misclassified_images[0].keys())

    for idx in range(num_images):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        img_data = misclassified_images[idx]
        image = img_data["image"].copy()  # Assume image is a numpy array

        # If image is in channel-first format (C, H, W) with 3 or 4 channels, transpose to (H, W, C).
        if image.ndim == 3 and image.shape[0] in (3, 4):
            image = np.transpose(image, (1, 2, 0))
        
        # Now image is expected to be (H, W, C)
        if image.ndim == 3:
            if image.shape[2] == 3:
                # Input order is (Green, Blue, Red); reassemble into (Red, Green, Blue)
                base_rgb = image[..., [2, 0, 1]]
                composite = base_rgb
            elif image.shape[2] == 4:
                # Input order is (Green, Blue, Gray, Red).
                # Construct base_rgb from indices: [3, 0, 1] meaning:
                #   Red comes from channel 3, Green from channel 0, Blue from channel 1.
                base_rgb = image[..., [3, 0, 1]]
                # Use channel index 2 as the overlay (gray channel)
                overlay = image[..., 2]
                composite = base_rgb
            else:
                composite = image  # Fallback if unexpected number of channels.
        else:
            composite = image

        # Normalize if desired. Apply per-channel normalization to base_rgb and overlay separately.
        if normalize:
            # For base RGB:
            if composite.ndim == 3:
                norm_base = np.zeros_like(composite, dtype=np.float32)
                for c in range(composite.shape[-1]):
                    ch = composite[..., c]
                    min_val = ch.min()
                    max_val = ch.max()
                    if max_val > min_val:
                        norm_base[..., c] = (ch - min_val) / (max_val - min_val)
                    else:
                        norm_base[..., c] = ch
                composite = norm_base
            # If there's an overlay (from a 4-channel image), normalize it too.
            if image.ndim == 3 and image.shape[2] == 4:
                min_val = overlay.min()
                max_val = overlay.max()
                if max_val > min_val:
                    overlay = (overlay - min_val) / (max_val - min_val)
        else:
            composite = composite / 255.0
            if image.ndim == 3 and image.shape[2] == 4:
                overlay = overlay / 255.0

        # Display: For 3-channel images, simply show the composite.
        # For 4-channel images, first show the composite (RGB) then overlay the normalized gray channel.
        if image.ndim == 3 and image.shape[2] == 4:
            ax.imshow(composite)
            ax.imshow(overlay, cmap='gray', alpha=overlay_alpha_DAPI)
        else:
            ax.imshow(composite)
        ax.axis('off')

        # Map predicted and true labels using class_names if provided.
        if class_names is not None:
            try:
                pred_label = class_names[int(img_data["pred_label"])]
                true_label = class_names[int(img_data["true_label"])]
            except Exception as e:
                print("Error mapping labels using class_names:", e)
                pred_label = img_data["pred_label"]
                true_label = img_data["true_label"]
        else:
            pred_label = img_data["pred_label"]
            true_label = img_data["true_label"]

        ax.set_title(f"True: {true_label}\nPred: {pred_label}")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()
    
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary classification results.
    The ROC curve shows the trade-off between the True Positive Rate (TPR) and 
    False Positive Rate (FPR) at various classification thresholds.
    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) binary labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the ROC curve plot.
    Notes
    -----
    The plot includes:
    - ROC curve in orange
    - Diagonal dashed line representing random chance
    - Area Under the Curve (AUC) score in the legend
    - Grid for better readability
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt.gcf()