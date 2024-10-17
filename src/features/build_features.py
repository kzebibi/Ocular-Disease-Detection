import cv2
import numpy as np
from skimage import exposure, filters, measure
from scipy.stats import entropy


def ensure_uint8(image):
    if image.dtype != np.uint8:
        return (image * 255).astype(np.uint8)
    return image


def apply_histogram_equalization(image):
    """
    Apply global histogram equalization to the image.
    """
    image = ensure_uint8(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert color image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # Image is already grayscale, no conversion needed
        image = image.squeeze()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.
    """
    image = ensure_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_adaptive_equalization(image, clip_limit=0.03):
    """
    Apply adaptive equalization using scikit-image's exposure module.
    """
    image = ensure_uint8(image)
    return (exposure.equalize_adapthist(image, clip_limit=clip_limit) * 255).astype(np.uint8)


def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to the image.
    """
    image = ensure_uint8(image)
    return exposure.adjust_gamma(image, gamma=gamma)


def apply_contrast_stretching(image, p2=2, p98=98):
    """
    Apply contrast stretching to the image.
    """
    image = ensure_uint8(image)
    p2, p98 = np.percentile(image, (p2, p98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))


def apply_unsharp_masking(image, radius=5, amount=1.0):
    """
    Apply unsharp masking to enhance image details.
    """
    image = ensure_uint8(image)
    # Remove the multichannel parameter and add channel_axis=None for grayscale images
    blurred = filters.gaussian(image, sigma=radius, channel_axis=None)
    return np.clip((image.astype(float) + amount * (image.astype(float) - blurred * 255)), 0, 255).astype(np.uint8)


def build_equalization_features(image):
    """
    Build a feature set of different equalization techniques.
    """
    # Check if the image is already grayscale
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # Image is already grayscale
        gray_image = image.squeeze()
    else:
        # Convert to grayscale if it's a color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply different equalization techniques
    hist_eq = apply_histogram_equalization(gray_image)
    clahe_eq = apply_clahe(gray_image)
    adaptive_eq = apply_adaptive_equalization(gray_image)
    # gamma_corrected = apply_gamma_correction(gray_image, gamma=0.8)
    # contrast_stretched = apply_contrast_stretching(gray_image)
    # unsharp_masked = apply_unsharp_masking(gray_image)

    # Include all 7 equalized images (including the original)
    selected_arrays = [gray_image, hist_eq, clahe_eq, adaptive_eq]

    # Ensure all arrays have the same shape
    shapes = [arr.shape for arr in selected_arrays]
    if len(set(shapes)) > 1:
        # If shapes are different, resize all to the smallest shape
        min_shape = min(shapes)
        selected_arrays = [cv2.resize(arr, min_shape[::-1]) for arr in selected_arrays]

    # Stack all equalized images
    equalized_features = np.stack(selected_arrays, axis=-1)

    return equalized_features


def shannon_entropy(image):
    """
    Calculate the Shannon entropy of an image.
    """
    histogram = np.histogram(image, bins=256, range=(0, 255))[0]
    histogram = histogram / np.sum(histogram)
    return entropy(histogram, base=2)


def extract_statistics(equalized_features):
    """
    Extract statistical features from the equalized images.
    """
    stats = []
    for i in range(equalized_features.shape[-1]):
        layer = equalized_features[..., i]
        stats.extend([
            np.mean(layer),
            np.std(layer),
            np.min(layer),
            np.max(layer),
            np.median(layer),
            np.percentile(layer, 25),
            np.percentile(layer, 75),
            shannon_entropy(layer),
        ])
    return np.array(stats)


def build_features(image):
    """
    Main function to build features from an input image.
    """
    equalized_features = build_equalization_features(image)
    statistical_features = extract_statistics(equalized_features)

    return equalized_features, statistical_features
