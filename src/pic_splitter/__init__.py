import os
import sys
import warnings

import numpy as np
from skimage import color, io, segmentation, filters, measure, morphology
from sklearn.cluster import KMeans

# Suppress low contrast warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image")


def felzenszwalb_segmentation(image: np.ndarray, scale: float = 100, min_size: int = 200) -> np.ndarray:
    """
    Apply Felzenszwalb segmentation algorithm.
    
    Args:
        image: RGB image
        scale: Higher scale means larger segments
        min_size: Minimum component size
        
    Returns:
        Segmentation labels
    """
    return segmentation.felzenszwalb(
        image, 
        scale=scale,
        sigma=0.5,
        min_size=min_size
    )


def slic_segmentation(image: np.ndarray, n_segments: int = 30) -> np.ndarray:
    """
    Apply SLIC segmentation to get superpixels.
    
    Args:
        image: RGB image
        n_segments: Target number of segments
        
    Returns:
        Segmentation labels
    """
    return segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=10,
        sigma=1,
        start_label=1
    )


def color_quantize(image: np.ndarray, n_colors: int = 8) -> np.ndarray:
    """
    Quantize the colors in an image to a specific number of colors.
    
    Args:
        image: RGB image
        n_colors: Number of colors to quantize to
        
    Returns:
        Quantized RGB image
    """
    # Reshape the image
    h, w, d = image.shape
    pixels = image.reshape((-1, d))
    
    # Convert to LAB for better color perception
    pixels_lab = color.rgb2lab(pixels)
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixels_lab)
    centers = kmeans.cluster_centers_
    
    # Convert centers back to RGB
    centers_rgb = color.lab2rgb(centers.reshape((-1, 1, 3))).reshape((-1, 3))
    
    # Create the quantized image
    quantized = np.zeros_like(pixels)
    for i in range(n_colors):
        quantized[labels == i] = centers_rgb[i]
    
    return quantized.reshape((h, w, d))


def texture_aware_segmentation(image: np.ndarray, n_colors: int = 8, texture_threshold: float = 0.2) -> np.ndarray:
    """
    Perform texture-aware segmentation using multiple algorithms.
    
    Args:
        image: RGB image
        n_colors: Number of colors for quantization
        texture_threshold: Threshold for texture similarity
        
    Returns:
        Segmentation labels
    """
    print("Applying multiple segmentation methods...")
    
    results = {}
    
    # Method 1: Simple Felzenszwalb segmentation
    for scale in [100, 200, 400]:
        labels = felzenszwalb_segmentation(image, scale=scale, min_size=50)
        num_segments = len(np.unique(labels))
        name = f"Felzenszwalb (scale={scale})"
        results[name] = (labels, num_segments)
        print(f"  - {name}: {num_segments} segments")
    
    # Method 2: SLIC segmentation with different segment counts
    for count in [15, 30, 60]:
        labels = slic_segmentation(image, n_segments=count)
        num_segments = len(np.unique(labels))
        name = f"SLIC (n={count})"
        results[name] = (labels, num_segments)
        print(f"  - {name}: {num_segments} segments")
    
    # Method 3: Color-based segmentation with k-means
    # Quantize colors first
    quantized = color_quantize(image, n_colors=n_colors)
    
    # Convert to LAB
    lab_image = color.rgb2lab(quantized)
    
    # Create a mask for each color cluster
    h, w, _ = lab_image.shape
    color_mask = np.zeros((h, w), dtype=np.int32)
    
    # Reshape and cluster again to get discrete regions
    pixels_lab = lab_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=min(15, n_colors*2), random_state=42, n_init="auto")
    color_labels = kmeans.fit_predict(pixels_lab)
    color_mask = color_labels.reshape((h, w))
    
    # Apply connected components to get contiguous regions
    labels = measure.label(color_mask, connectivity=2)
    num_segments = len(np.unique(labels))
    name = f"Color-based (n_colors={n_colors})"
    results[name] = (labels, num_segments)
    print(f"  - {name}: {num_segments} segments")
    
    # Find best method with target number of segments (10-40)
    best_method = None
    best_count = 0
    target_segments = 25  # Aim for this many segments
    
    for name, (labels, count) in results.items():
        if 10 <= count <= 40:
            if not best_method or abs(count - target_segments) < abs(best_count - target_segments):
                best_method = name
                best_count = count
    
    # If no method with 10-40 segments, find closest to target
    if not best_method:
        for name, (labels, count) in results.items():
            if not best_method or abs(count - target_segments) < abs(best_count - target_segments):
                best_method = name
                best_count = count
    
    print(f"\nSelected: {best_method} with {best_count} segments")
    
    # Use the selected segmentation method
    final_labels = results[best_method][0]
    
    # Clean up small regions
    min_region_size = 50
    cleaned_labels = morphology.remove_small_objects(final_labels, min_size=min_region_size)
    cleaned_labels = measure.label(cleaned_labels)
    
    final_segment_count = len(np.unique(cleaned_labels))
    print(f"Final segment count: {final_segment_count}")
    
    return cleaned_labels


def process_image(image_path: str, n_colors: int = 8, texture_threshold: float = 0.2) -> None:
    """
    Process an image and split it into segments based on color and texture patterns.
    
    Args:
        image_path: Path to the input image
        n_colors: Number of colors to quantize the image to
        texture_threshold: Threshold for texture similarity (higher = more merging)
    """
    # Create output directory in the current working directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.getcwd(), f"{base_name}_segments")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load image
    print("Loading image...")
    image = io.imread(image_path)
    
    # Apply segmentation
    print(f"Segmenting with {n_colors} colors...")
    
    labels = texture_aware_segmentation(
        image, 
        n_colors=n_colors,
        texture_threshold=texture_threshold
    )
    
    num_segments = len(np.unique(labels)) - 1  # Subtract 1 for background
    print(f"\nFound {num_segments} segments")
    
    # Process and save each segment
    print("\nSaving segments...")
    saved_count = 0
    
    # Get segment properties for sorting
    segment_info = {}
    for label in np.unique(labels):
        if label == 0:  # Skip background
            continue
            
        mask = labels == label
        size = np.sum(mask)
        
        # Skip very small segments
        if size < 50:
            continue
            
        # Calculate mean color
        mean_color = np.mean(image[mask], axis=0)
        # Calculate brightness for sorting
        brightness = 0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]
        
        segment_info[label] = {
            "size": size,
            "brightness": brightness
        }
    
    # Sort segments by size (largest first)
    sorted_segments = sorted(segment_info.items(), key=lambda x: x[1]["size"], reverse=True)
    
    for label, info in sorted_segments:
        mask = labels == label
        
        # Create RGBA image with transparency
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Copy the image pixels where the mask is True
        for c in range(3):  # RGB channels
            rgba_image[:,:,c][mask] = image[:,:,c][mask]
            
        # Set alpha channel
        rgba_image[:,:,3][mask] = 255  # Fully opaque where segment exists
        
        # Save the segment
        segment_path = os.path.join(output_dir, f"segment_{saved_count:03d}.png")
        io.imsave(segment_path, rgba_image)
        saved_count += 1
    
    print(f"\nProcessing complete. {saved_count} segmented images saved to {output_dir}/")


def main() -> None:
    """Main entry point for the pic-splitter command."""
    if len(sys.argv) < 2:
        print("Usage: pic-splitter <image_path> [n_colors] [texture_threshold]")
        print("  - n_colors: Number of colors to quantize to (4-16, default: 8)")
        print("  - texture_threshold: Texture similarity threshold (0.1-1.0, default: 0.2)")
        print("                       Higher values merge more regions with different textures")
        print("\nExamples:")
        print("  pic-splitter image.jpg                # Default (8 colors, 0.2 threshold)")
        print("  pic-splitter image.jpg 6              # Use 6 colors")
        print("  pic-splitter image.jpg 10 0.3         # Use 10 colors and more texture merging")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found")
        sys.exit(1)
    
    # Parse optional parameters
    n_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    texture_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    # Validate inputs
    n_colors = max(4, min(16, n_colors))
    texture_threshold = max(0.1, min(1.0, texture_threshold))
    
    process_image(
        image_path,
        n_colors=n_colors,
        texture_threshold=texture_threshold
    )