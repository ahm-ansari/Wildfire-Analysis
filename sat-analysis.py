import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings from KMeans about memory leaks on Windows with MKL.
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

def perform_landscape_analysis(image_path="sd-3layers1.jpg"):
    """
    Performs EDA and unsupervised clustering on a geospatial image.
    The image channels are interpreted as: Red=Altitude, Green=Slope, Blue=Aspect.
    """
    # =========================================================================
    # 1. Load the Data
    # =========================================================================
    try:
        print(f"1. Loading data from '{image_path}'...")
        img = Image.open(image_path)
        # Convert the image to a NumPy array. Shape: (height, width, 3)
        image_data = np.array(img)
        img.show(title="Original Image")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please make sure the image file is in the same directory as the script.")
        return
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    # =========================================================================
    # 2. Data Preprocessing and Exploratory Data Analysis (EDA)
    # =========================================================================
    print("\n2. Preprocessing data and performing EDA...")

    # Reshape the data for ML: from a 3D image array to a 2D array of pixels
    # Each row is a pixel, and columns are Altitude, Slope, Aspect
    h, w, c = image_data.shape
    pixel_features = image_data.reshape(-1, c)
    print(f"   - Image dimensions: {h}x{w} pixels")
    print(f"   - Reshaped data for ML: {pixel_features.shape[0]} samples, {pixel_features.shape[1]} features")

    # EDA: Plot histograms for each feature (channel)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channel_names = ['Altitude (Red)', 'Slope (Green)', 'Aspect (Blue)']
    colors = ['red', 'green', 'blue']

    for i, name in enumerate(channel_names):
        axes[i].hist(pixel_features[:, i], bins=50, color=colors[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {name}')
        axes[i].set_xlabel('Pixel Intensity (0-255)')
        axes[i].set_ylabel('Frequency')

    plt.suptitle('Exploratory Data Analysis: Feature Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Feature Scaling: Standardize features to have a mean of 0 and variance of 1.
    # This is crucial for distance-based algorithms like K-Means.
    print("   - Scaling features using StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pixel_features)

    # =========================================================================
    # 3. Identify the ML Model
    # =========================================================================
    print("\n3. Identifying ML Model...")
    # For unlabeled data, clustering is the appropriate task.
    # We will use K-Means, a popular and effective clustering algorithm.
    # We'll choose to find 5 distinct clusters in the landscape.
    n_clusters = 5
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    print(f"   - Model: K-Means Clustering")
    print(f"   - Goal: Group pixels into {n_clusters} distinct landscape types.")

    # =========================================================================
    # 4. Train and "Test" the Model
    # =========================================================================
    # In unsupervised learning, "training" is fitting the model to the data to find the clusters.
    # We will fit it on the entire dataset to segment the whole image.
    print("\n4. Training the K-Means model...")
    model.fit(scaled_features)
    print("   - Model training complete.")

    # The "test" phase is to apply the learned cluster labels to our data.
    cluster_labels = model.labels_
    print(f"   - Assigned each of the {len(cluster_labels)} pixels to a cluster.")

    # =========================================================================
    # 5. "Deploy" the Model (Visualize the Results)
    # =========================================================================
    # In this context, "deploying" means using the trained model to create a
    # new, segmented map showing the different landscape clusters.
    print("\n5. Deploying model to generate segmented landscape map...")

    # Reshape the labels back into the original image dimensions
    segmented_image_data = cluster_labels.reshape(h, w)

    # Display the original image and the resulting cluster map
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img)
    axes[0].set_title('Original Feature Image\n(R:Alt, G:Slope, B:Aspect)')
    axes[0].axis('off')

    # Use a colormap for clear visualization of the clusters
    im = axes[1].imshow(segmented_image_data, cmap='viridis')
    axes[1].set_title(f'Segmented Landscape ({n_clusters} Clusters)')
    axes[1].axis('off')
    
    # Add a color bar to explain the cluster labels
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(n_clusters))
    cbar.set_label('Cluster ID')

    plt.suptitle('Machine Learning: Unsupervised Landscape Segmentation', fontsize=18)
    plt.show()
    print("\nAnalysis complete. The segmented map shows the distinct regions identified by the model.")


if __name__ == "__main__":
    # Make sure 'sd-3layers.jpg' is in the same folder as this script
    perform_landscape_analysis(image_path="sd-3layers.jpg")
