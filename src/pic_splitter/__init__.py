import os
import sys

import numpy as np
from skimage import color, graph, io, measure, segmentation
from skimage.filters import sobel, threshold_otsu


def process_image(image_path: str) -> None:
    """
    Process an image and split it into segments based on color and texture.
    Optimized for abstract images with distinct regions.

    Args:
        image_path: Path to the input image
    """
    # Create output directory named after input file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"{base_name}_segments"
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = io.imread(image_path)

    # Method 1: Color-based segmentation using quickshift
    segments_color = segmentation.quickshift(
        image,
        kernel_size=5,  # Smaller kernel size for more detailed segments
        max_dist=10,  # Smaller max distance for tighter clusters
        ratio=0.5,  # Balance between color and spatial proximity
    )

    # Method 2: Edge-based segmentation
    # Convert to grayscale and compute edges
    gray = color.rgb2gray(image)
    edges = sobel(gray)

    # Threshold edges to create binary image
    threshold = threshold_otsu(edges)
    binary_edges = edges > threshold

    # Watershed segmentation using edges
    markers = measure.label(binary_edges)
    segments_edge = segmentation.watershed(edges, markers)

    # Method 3: SLIC superpixels with normalized cuts
    segments_slic = segmentation.slic(
        image, n_segments=100, compactness=10, sigma=1, start_label=1
    )

    # Build Region Adjacency Graph
    rag = graph.rag_mean_color(image, segments_slic)

    # Cut graph to get final segmentation
    segments_graph = graph.cut_normalized(segments_slic, rag)

    # Create a dictionary to store segment count for each method
    segment_counts = {
        "color": len(np.unique(segments_color)),
        "edge": len(np.unique(segments_edge)),
        "graph": len(np.unique(segments_graph)),
    }

    # Choose the method with most segments
    best_method = max(segment_counts.items(), key=lambda x: x[1])
    print("Segmentation methods comparison:")
    print(f"  - Color-based (quickshift): {segment_counts['color']} segments")
    print(f"  - Edge-based (watershed): {segment_counts['edge']} segments")
    print(f"  - Graph-based (SLIC+normalized cuts): {segment_counts['graph']} segments")
    print(f"Using {best_method[0]}-based method with {best_method[1]} segments")

    if best_method[0] == "color":
        labels = segments_color
    elif best_method[0] == "edge":
        labels = segments_edge
    else:
        labels = segments_graph

    # Process and save each segment
    saved_count = 0
    min_size = 100  # Minimum number of pixels to save a segment

    for label in np.unique(labels):
        mask = labels == label

        # Skip segments that are too small
        if np.sum(mask) < min_size:
            continue

        # Create segmented image
        segmented = np.zeros_like(image)
        segmented[mask] = image[mask]

        # Create masked image (only segment on transparent background)
        masked = image.copy()
        if masked.shape[2] == 3:  # Convert RGB to RGBA if needed
            alpha = (
                np.ones((masked.shape[0], masked.shape[1], 1), dtype=masked.dtype) * 255
            )
            masked = np.concatenate([masked, alpha], axis=2)

        # Set alpha to 0 for background
        masked[~mask, 3] = 0

        # Save the segment
        segment_path = os.path.join(output_dir, f"segment_{saved_count:03d}.png")
        io.imsave(segment_path, masked)
        saved_count += 1

    print(f"Processing complete. {saved_count} segmented images saved to {output_dir}/")


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
