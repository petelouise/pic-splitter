import os
import sys
import warnings

import numpy as np
from skimage import color, io, segmentation

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


def process_image(image_path: str) -> None:
    """
    Process an image and split it into segments based on color.
    
    Args:
        image_path: Path to the input image
    """
    # Create output directory named after input file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"{base_name}_segments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print("Loading image...")
    image = io.imread(image_path)
    
    # Apply segmentation with different parameters
    print("Testing different segmentation methods...")
    
    results = {}
    
    # Try Felzenszwalb with different scales
    scales = [400, 800, 1600]  # Use higher scales for fewer segments
    for scale in scales:
        labels = felzenszwalb_segmentation(image, scale=scale, min_size=200)
        num_segments = len(np.unique(labels))
        name = f"Felzenszwalb (scale={scale})"
        results[name] = (labels, num_segments)
        print(f"  - {name}: {num_segments} segments")
    
    # Try SLIC with different segment counts
    segment_counts = [20, 30, 40]
    for count in segment_counts:
        labels = slic_segmentation(image, n_segments=count)
        num_segments = len(np.unique(labels))
        name = f"SLIC (n={count})"
        results[name] = (labels, num_segments)
        print(f"  - {name}: {num_segments} segments")
    
    # Find best method with 15-40 segments
    best_method = None
    best_count = 0
    
    for name, (labels, count) in results.items():
        # Prefer methods with 15-40 segments
        if 15 <= count <= 40:
            # Choose the one closest to 25 segments
            if not best_method or abs(count - 25) < abs(best_count - 25):
                best_method = name
                best_count = count
    
    # If no method with 15-40 segments, find closest to 25
    if not best_method:
        for name, (labels, count) in results.items():
            if not best_method or abs(count - 25) < abs(best_count - 25):
                best_method = name
                best_count = count
    
    print(f"\nSelected: {best_method} with {best_count} segments")
    
    # Use the selected segmentation method
    labels = results[best_method][0]
    
    # Process and save each segment
    print("\nSaving segments...")
    saved_count = 0
    min_size = 100  # Minimum number of pixels to save a segment
    
    # Debug the labels
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Image shape: {image.shape}")
    
    # Get segment properties
    segment_info = {}
    for label in np.unique(labels):
        mask = labels == label
        size = np.sum(mask)
        
        print(f"Label {label}: {size} pixels")
        
        # Skip small segments
        if size < min_size:
            print(f"  - Skipping (too small)")
            continue
            
        # Calculate mean color
        mean_color = np.mean(image[mask], axis=0)
        # Calculate brightness for sorting
        brightness = 0.299 * mean_color[0] + 0.587 * mean_color[1] + 0.114 * mean_color[2]
        
        segment_info[label] = {
            "size": size,
            "brightness": brightness
        }
    
    print(f"Found {len(segment_info)} segments after filtering")
    
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
        print(f"Saving segment {saved_count} (label {label}, {info['size']} pixels)")
        io.imsave(segment_path, rgba_image)
        saved_count += 1
    
    print(f"\nProcessing complete. {saved_count} segmented images saved to {output_dir}/")


def main() -> None:
    """Main entry point for the pic-splitter command."""
    if len(sys.argv) < 2:
        print("Usage: pic-splitter <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found")
        sys.exit(1)
    
    process_image(image_path)