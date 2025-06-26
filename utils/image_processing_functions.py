import numpy as np
import torch

def extract_images_features(
    images_paths, 
    feature_extractor,
    transforms,
    device=None
):
    """
    Extract features from images using a given feature extractor and transforms
    
    Args:
        images_paths (list): List of paths to images
        labels (list): List of corresponding labels
        feature_extractor (torch.nn.Module): Neural network for feature extraction
        transforms (monai.transforms.Compose): Composition of image transforms
        device (torch.device): Device to run extraction on
        
    Returns:
        np.ndarray: Extracted features array of shape (n_images, n_features)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    features_list = []
    #create a list of dictionaries with the image path so that loadImaged works
    train_data = [{"image": img} for img in images_paths] 
    
    for data in train_data:
        transformed = transforms(data)
        img_tensor = transformed["image"].to(device)
        img_tensor = img_tensor.unsqueeze(0)
        #print(f"image shape: {img_tensor.shape}")
        
        with torch.no_grad():
            feat = feature_extractor(img_tensor)
        features_list.append(feat.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    print(f"Extracted features shape: {features.shape}")
    
    return features