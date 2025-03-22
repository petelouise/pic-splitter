import os
import sys
import warnings

import numpy as np
from skimage import color, io, segmentation, measure, morphology, filters
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

# Suppress low contrast warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image")


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
    w, h, d = image.shape
    pixels = image.reshape((w * h, d))
    
    # Convert to LAB for better color perception
    pixels_lab = color.rgb2lab(pixels.reshape((-1, 1, 3))).reshape((-1, 3))
    
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
    
    return quantized.reshape((w, h, d))


def texture_aware_segmentation(image: np.ndarray, n_colors: int = 8, texture_threshold: float = 0.2) -> np.ndarray:
    """
    Perform texture-aware segmentation that respects both color and texture patterns.
    
    Args:
        image: RGB image
        n_colors: Number of colors for quantization
        texture_threshold: Threshold for texture similarity
        
    Returns:
        Segmentation labels
    """
    # Step 1: Quantize colors
    print("Step 1: Quantizing colors...")
    quantized = color_quantize(image, n_colors=n_colors)
    
    # Step 2: Initial segmentation based on color regions
    print("Step 2: Creating initial segments based on color...")
    # Convert to LAB for better color discrimination
    lab_image = color.rgb2lab(quantized)
    
    # Create an initial segmentation based on color
    initial_segments = np.zeros((lab_image.shape[0], lab_image.shape[1]), dtype=np.int32)
    
    # We'll use a region growing approach where regions with similar colors are grouped
    # First convert the LAB image to a set of unique color indices
    reshaped_lab = lab_image.reshape((-1, 3))
    unique_colors = np.unique(reshaped_lab, axis=0)
    color_to_index = {tuple(color): i+1 for i, color in enumerate(unique_colors)}
    
    # Assign each pixel to its color index
    for i in range(lab_image.shape[0]):
        for j in range(lab_image.shape[1]):
            pixel_color = tuple(lab_image[i, j])
            # Find the closest color in our unique colors (within a small threshold)
            min_dist = float('inf')
            closest_idx = None
            for color_idx, color in enumerate(unique_colors):
                dist = np.linalg.norm(np.array(pixel_color) - color)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = color_idx + 1
            initial_segments[i, j] = closest_idx
    
    # Step 3: Apply connected components to get contiguous regions
    print("Step 3: Finding connected regions...")
    labels = measure.label(initial_segments, connectivity=2)
    
    # Step 4: Analyze texture within each region
    print("Step 4: Analyzing texture patterns...")
    gray = color.rgb2gray(image)
    
    # Apply Gabor filters for texture analysis
    texture_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        # Real part of Gabor filter response
        real_gabor = filters.gabor(gray, frequency=0.6, theta=theta)[0]
        texture_features.append(real_gabor)
    
    # Stack features
    texture_stack = np.stack(texture_features, axis=-1)
    
    # Step 5: Merge regions with similar texture and color
    print("Step 5: Merging regions with similar texture...")
    
    # We'll create a region adjacency graph
    regions = measure.regionprops(labels)
    n_regions = len(regions)
    
    # For each region, compute average color and texture
    region_stats = {}
    for region in regions:
        label = region.label
        mask = labels == label
        
        # Skip tiny regions
        if np.sum(mask) < 50:
            continue
            
        # Compute mean color in LAB space
        mean_color = np.mean(lab_image[mask], axis=0)
        
        # Compute texture statistics (mean and variance of filter responses)
        texture_means = []
        texture_vars = []
        for i in range(texture_stack.shape[-1]):
            texture_means.append(np.mean(texture_stack[mask, i]))
            texture_vars.append(np.var(texture_stack[mask, i]))
        
        texture_features = np.array(texture_means + texture_vars)
        
        region_stats[label] = {
            'color': mean_color,
            'texture': texture_features,
            'size': np.sum(mask),
            'bbox': region.bbox,
            'centroid': region.centroid
        }
    
    # Find adjacent regions
    from scipy.spatial import cKDTree
    centroids = np.array([region_stats[label]['centroid'] for label in region_stats])
    kdtree = cKDTree(centroids)
    
    # For each region, find its neighbors
    merges = []
    for label1, stats1 in region_stats.items():
        # Find potential neighbors (regions whose centroids are close)
        neighbors = kdtree.query_ball_point(stats1['centroid'], r=50)
        
        for idx in neighbors:
            label2 = list(region_stats.keys())[idx]
            if label1 >= label2:
                continue
                
            stats2 = region_stats[label2]
            
            # Compute color similarity (Euclidean distance in LAB space)
            color_diff = np.linalg.norm(stats1['color'] - stats2['color'])
            
            # Compute texture similarity
            texture_diff = np.linalg.norm(stats1['texture'] - stats2['texture'])
            
            # If both color and texture are similar, consider merging
            if color_diff < 10 and texture_diff < texture_threshold:
                # Add to merge list
                merges.append((label1, label2))
    
    # Apply merges to create final segmentation
    print(f"Planning to merge {len(merges)} region pairs...")
    
    final_labels = labels.copy()
    merged = set()
    
    for label1, label2 in merges:
        if label1 in merged or label2 in merged:
            continue
            
        # Merge label2 into label1
        final_labels[final_labels == label2] = label1
        merged.add(label2)
    
    # Relabel to ensure sequential labels
    final_labels = measure.label(final_labels > 0)
    
    # Step 6: Final cleanup - remove small regions
    print("Step 6: Final cleanup...")
    
    # Remove very small regions
    min_size = 100  # Minimum region size
    final_labels = morphology.remove_small_objects(final_labels, min_size=min_size)
    final_labels = measure.label(final_labels)
    
    return final_labels


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
    print(f"Segmenting with {n_colors} colors and texture threshold {texture_threshold}...")
    
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