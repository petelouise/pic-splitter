import sys

import numpy as np
from skimage import color, feature, graph, io, segmentation
from skimage.util import img_as_ubyte


def process_image(image_path: str) -> None:
    # Load image
    image = io.imread(image_path)
    gray = color.rgb2gray(image)  # Convert to grayscale for texture analysis

    # Compute Local Binary Pattern (LBP) for texture detection
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")

    # Normalize LBP for consistent scaling
    lbp = img_as_ubyte(lbp / np.max(lbp))

    # Generate superpixels
    segments = segmentation.slic(image, n_segments=250, compactness=10, start_label=1)

    # Compute mean color & texture for each superpixel
    def compute_region_features(region_labels, img, texture):
        features = {}
        for region in np.unique(region_labels):
            mask = region_labels == region
            mean_color = np.mean(img[mask], axis=0)
            mean_texture = np.mean(texture[mask])
            features[region] = np.hstack((mean_color, mean_texture))
        return features

    color_features = compute_region_features(segments, image, lbp)

    # Build Region Adjacency Graph (RAG) using color & texture similarity
    rag = graph.rag_mean_color(image, segments, mode="similarity")

    # Add texture similarity to edge weights
    for edge in rag.edges:
        n1, n2 = edge
        texture_diff = abs(color_features[n1][-1] - color_features[n2][-1])
        rag[n1][n2][
            "weight"
        ] += texture_diff  # Integrate texture into similarity metric

    # Merge regions using combined color + texture similarity
    labels = graph.cut_threshold(segments, rag, thresh=40)

    # Save each segmented object separately
    for label in np.unique(labels):
        mask = labels == label
        segmented = image.copy()
        segmented[~mask] = 0
        io.imsave(f"segment_{label}.png", segmented)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pic_splitter.py <image_path>")
        sys.exit(1)
    process_image(sys.argv[1])
    print("Processing complete. Segmented images saved.")
