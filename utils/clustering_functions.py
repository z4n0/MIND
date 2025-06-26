from collections import Counter
from typing import List, Dict, Union
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_cluster_image_names(image_paths: Union[List[str], np.ndarray], cluster_labels: Union[List[int], np.ndarray]) -> Dict[int, List[str]]:
    """
    Prints the file names (last part of the path) of images in each cluster.
    
    Args:
        image_paths (list or array): List of file paths to the images.
        cluster_labels (list or array): Cluster label for each image.
        
    Returns: 
        dict: Dictionary with cluster labels as keys and lists of file names as values
    """
    # Group the file names by cluster label
    clusters = {}
    for path, label in zip(image_paths, cluster_labels):
        filename = os.path.basename(path)
        #setdefault returns the value of the key if it is in the dictionary, if not it inserts the key with the specified value
        clusters.setdefault(label, []).append(filename)
    
    return clusters

def compute_cluster_mapping(true_labels: Union[List[int], np.ndarray], cluster_labels: Union[List[int], np.ndarray]) -> Dict[int, int]:
    """
    For each cluster, find the majority true label.
    it takes the cluster_labels to identify the true_labels associated to each cluster
    take the majority true_label as the label of the cluster.
    
    Args:
        true_labels (list/numpy.ndarray): Array of true labels for the data points.
        cluster_labels (list/numpy.ndarray): Array of cluster labels assigned by the clustering algorithm.
        
    Returns:
        dict: A dictionary mapping each cluster label to the majority true label.
    """
    cluster_to_class = {}
    for cl in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cl)[0] # indices of data points in cluster cl
        # Count the occurrences of each true label in the cluster and find the majority label
        majority = Counter([true_labels[i] for i in indices]).most_common(1)[0][0]
        cluster_to_class[cl] = majority
    return cluster_to_class

def get_color(true_label: int, predicted_label: int) -> str:
    """
    Determine the color based on the true and predicted labels.

    Args:
        true_label (int): The true label of the data point.
        predicted_label (int): The predicted label of the data point.

    Returns:
        str: The color representing the classification result.
    """
    # Define correct color mapping:
    #   True label 0 (MSAP) → blue, True label 1 (PD) → red.
    correct_color_map = {0: "blue", 1: "red"}
    if true_label == predicted_label:
        return correct_color_map[true_label]
    else:
        # Misclassification: define special colors.
        if true_label == 0 and predicted_label == 1:
            return "orange"  # class0 misclassified as class1. 
        elif true_label == 1 and predicted_label == 0:
            return "cyan"   # class1 misclassified as class0.
        else:
            return "gray"

     
def perform_clustering(features_list, true_labels, n_clusters=2, class0_name="Class 0", class1_name="Class 1"):
    """
    Perform K-Means clustering on the given features and map the resulting clusters to true classes.
    Parameters:
    features_list (array-like): A list or array of feature vectors to cluster.
    true_labels (array-like): The true labels corresponding to the feature vectors.
    n_clusters (int, optional): The number of clusters to form. Default is 2.
    class0_name (str, optional): The name for class 0. Default is "Class 0".
    class1_name (str, optional): The name for class 1. Default is "Class 1".
    Returns:
    tuple: A tuple containing:
        - raw_cluster_labels (array): The raw cluster labels assigned by K-Means.
        - predicted_labels (array): The predicted labels after mapping clusters to true classes.
        - cluster_to_class (dict): A dictionary mapping each cluster to the corresponding true class.
    """
    from sklearn.cluster import KMeans  
    # Step 1: K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    raw_cluster_labels = kmeans.fit_predict(features_list)
    
    # Step 2: Map each cluster to the true class via majority vote
    cluster_to_class = compute_cluster_mapping(true_labels, raw_cluster_labels)
    
    # Use the mapping to reassign predicted labels:
    predicted_labels = np.array([cluster_to_class[cl] for cl in raw_cluster_labels])
    
    return raw_cluster_labels, predicted_labels, cluster_to_class


from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull

def plot_clusters_2d(features_list: Union[List[List[float]], np.ndarray], 
                    true_labels: Union[List[int], np.ndarray], 
                    raw_cluster_labels: Union[List[int], np.ndarray], 
                    predicted_labels: Union[List[int], np.ndarray], 
                    #cluster_to_class_mapping: Dict[int, int], 
                    class0_name: str = "Class 0", 
                    class1_name: str = "Class 1") -> None:
    """
    Visualizes clusters in 2D using PCA, highlighting cluster regions and coloring points 
    based on true vs. predicted labels.

    Args:
        features_list (List[List[float]] or np.ndarray): Feature matrix.
        true_labels (List[int] or np.ndarray): Ground truth labels.
        raw_cluster_labels (List[int] or np.ndarray): Cluster assignments from the clustering algorithm.
        predicted_labels (List[int] or np.ndarray): Predicted labels after mapping clusters to classes.
        cluster_to_class_mapping (Dict[int, int]): Mapping from cluster ID to class label.
        class0_name (str, optional): Name for class 0 in the legend. Defaults to "Class 0".
        class1_name (str, optional): Name for class 1 in the legend. Defaults to "Class 1".
    
    Returns:
        None
    """
    # Step 4: Visualize in 2D using PCA
    pca_2d = PCA(n_components=2, random_state=42)
    features_pca2d = pca_2d.fit_transform(features_list)
    
    # Define colors based on correctness
    def get_color(true: int, pred: int) -> str:
        """
        Returns a color based on whether the true and predicted labels match.
        """
        if true == 0 and pred == 0:
            return "blue"
        elif true == 1 and pred == 1:
            return "red"
        elif true == 0 and pred == 1:
            return "orange"
        else:
            return "cyan"
    final_point_colors = [get_color(t, p) for t, p in zip(true_labels, predicted_labels)]

    # Assign distinct colors to each cluster using a colormap
    unique_clusters = np.unique(raw_cluster_labels)
    cmap = plt.get_cmap('tab10', len(unique_clusters))
    cluster_colors = {cl: cmap(i) for i, cl in enumerate(unique_clusters)}

    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Draw semi-transparent cluster regions using convex hulls
    for cl in unique_clusters:
        indices = np.where(raw_cluster_labels == cl)[0]
        if len(indices) < 3:
            continue  # Skip clusters with insufficient points
        
        cluster_points = features_pca2d[indices, :]
        try:
            hull = ConvexHull(cluster_points)
            hull_vertices = cluster_points[hull.vertices, :]
            poly = Polygon(hull_vertices, closed=True, facecolor=cluster_colors[cl], alpha=0.2, edgecolor='none')
            plt.gca().add_patch(poly)
        except:
            pass  # Handle convex hull errors silently

    # Plot points with true label colors
    plt.scatter(features_pca2d[:, 0], features_pca2d[:, 1], 
                c=final_point_colors, alpha=0.7, edgecolors='w', linewidths=0.5)

    # Create combined legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f"{class0_name} (correct)", 
               markerfacecolor="blue", markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f"{class1_name} (correct)", 
               markerfacecolor="red", markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f"{class0_name} misclassified", 
               markerfacecolor="orange", markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f"{class1_name} misclassified", 
               markerfacecolor="cyan", markersize=10)
    ]
    
    # Add cluster region legend entries
    for cl in unique_clusters:
        legend_elements.append(Patch(facecolor=cluster_colors[cl], alpha=0.2, 
                                   label=f'Cluster {cl} Region'))

    plt.legend(handles=legend_elements, fontsize='small', markerscale=0.7, loc='best')
    plt.title("PCA 2D Visualization with Cluster Regions")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()
    
def plot_clusters_2dx(features_list, true_labels, raw_cluster_labels, predicted_labels, class0_name="Class 0", class1_name="Class 1"):
    """
    Plot clusters with background colors and points colored by their true labels.
    perform PCA to reduce dimensionality to 2D.
    
    Args:
        features_list: Original feature matrix
        true_labels: Ground truth labels
        raw_cluster_labels: Cluster assignments
        predicted_labels: Model predictions
        cluster_to_class_mapping: Dictionary mapping cluster IDs to predicted classes
        class0_name: Name for class 0 (default: "Class 0")
        class1_name: Name for class 1 (default: "Class 1")
    """
    # Reduce dimensionality to 2D using PCA
    pca_2d = PCA(n_components=2, random_state=42)
    features_pca2d = pca_2d.fit_transform(features_list)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Define cluster background colors (using more visible pastel colors)
    cluster_colors = ['#FFB3B3', '#B3FFB3', '#B3B3FF', '#FFFFA8', '#FFB3FF', '#B3FFFF']
    
    # Plot cluster backgrounds using convex hull
    from scipy.spatial import ConvexHull
    
    for i, cluster_id in enumerate(np.unique(raw_cluster_labels)):
        # Get points belonging to the cluster
        mask = raw_cluster_labels == cluster_id
        cluster_points = features_pca2d[mask]
        
        if len(cluster_points) > 2:  # Need at least 3 points for ConvexHull
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            
            # Add padding to hull
            centroid = np.mean(hull_points, axis=0)
            hull_points = hull_points + (hull_points - centroid) * 0.1
            
            # Fill the hull with cluster color
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                    color=cluster_colors[i % len(cluster_colors)], 
                    alpha=0.3, 
                    label=f'Cluster {cluster_id}')
    
    # Plot points with colors based on true labels
    point_colors = ['#000000', '#0000FF']  # Black for class 0, Blue for class 1
    for label in [0, 1]:
        mask = true_labels == label
        plt.scatter(features_pca2d[mask, 0], features_pca2d[mask, 1],
                   c=[point_colors[label]], 
                   alpha=0.7,
                   label=f'{class0_name if label == 0 else class1_name}')
    
    # Add misclassification markers
    misclassified = true_labels != predicted_labels
    if np.any(misclassified):
        plt.scatter(features_pca2d[misclassified, 0], features_pca2d[misclassified, 1],
                   marker='x', c='red', s=100, alpha=0.7,
                   label='Misclassified')
    
    # Customize plot
    plt.title("Cluster Visualization with True Labels")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    # Create legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cropping
    plt.tight_layout()
    plt.show()
    

    
def evaluate_cluster(true_labels, raw_cluster_labels,class0_name="Class 0", class1_name="Class 1"):
    """
    Evaluate clustering performance using multiple metrics.
    
    Args:
        true_labels (array-like): Ground truth labels
        raw_cluster_labels (array-like): Cluster assignments from clustering algorithm
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
    import seaborn as sns
    
    # Convert inputs to numpy arrays
    true_labels = np.array(true_labels)
    raw_cluster_labels = np.array(raw_cluster_labels)

    # Calculate ARI and NMI
    ari = adjusted_rand_score(true_labels, raw_cluster_labels) # Adjusted Rand Index, ranges from -1 to 1, where 0 is random and 1 is a perfect match.
    nmi = normalized_mutual_info_score(true_labels, raw_cluster_labels) # Normalized Mutual Information, ranges from 0 to 1, where 0 means no mutual information and 1 means perfect correlation.
    
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
    
    # Get predicted labels using cluster mapping
    cluster_to_class = compute_cluster_mapping(true_labels, raw_cluster_labels)
    predicted_labels = np.array([cluster_to_class[cl] for cl in raw_cluster_labels])
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class0_name, class1_name],
                yticklabels=[class0_name, class1_name])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix: True Labels vs. Mapped Predicted Labels")
    plt.show()
    
    # Return metrics as dictionary
    return {
        "ari": ari,
        "nmi": nmi,
        "confusion_matrix": cm,
        "cluster_mapping": cluster_to_class,
        "predicted_labels": predicted_labels
    }