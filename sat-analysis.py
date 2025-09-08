import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def create_mock_wildfire_image(filename="sd-4layers.jpg", width=500, height=500):
    """
    Generates a mock satellite image with vegetation, scorched earth, and fire.
    """
    # Define colors
    VEGETATION_COLOR = (34, 139, 34)   # Forest Green
    SCORCHED_COLOR = (139, 69, 19)     # Saddle Brown
    FIRE_COLOR_BRIGHT = (255, 165, 0)  # Orange
    FIRE_COLOR_INTENSE = (255, 69, 0)  # Red-Orange

    # Create a base image with vegetation
    img = Image.new('RGB', (width, height), VEGETATION_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw a large "scorched" area
    draw.ellipse([50, 50, 400, 450], fill=SCORCHED_COLOR)

    # Draw "active fire" areas
    draw.ellipse([150, 180, 250, 280], fill=FIRE_COLOR_BRIGHT)
    draw.ellipse([230, 250, 330, 350], fill=FIRE_COLOR_INTENSE)
    draw.ellipse([200, 320, 280, 400], fill=FIRE_COLOR_BRIGHT)

    print(f"Generated mock satellite image: '{filename}'")
    img.save(filename)
    return filename

def analyze_wildfire_image(image_path):
    """
    Analyzes a satellite image to detect fire and estimate its extent.
    """
    try:
        # 1. Load the image using Pillow
        img = Image.open(image_path).convert('RGB')
        
        # 2. Convert the image to a NumPy array for numerical analysis
        # The array will have the shape (height, width, 3) for RGB channels
        image_data = np.array(img)
        
        print(f"\nImage loaded successfully. Shape of NumPy array: {image_data.shape}")

        # 3. Define color thresholds for analysis
        # These values would be tuned for real satellite imagery (e.g., infrared bands)
        # For this example, we define a "fire" color as high Red, moderate Green, low Blue
        # [R, G, B]
        lower_fire_threshold = np.array([200, 50, 0])
        upper_fire_threshold = np.array([256, 170, 50])

        # 4. Create a boolean mask for fire pixels
        # The mask will be `True` where pixels are within the fire color range
        fire_mask = np.all((image_data >= lower_fire_threshold) & (image_data <= upper_fire_threshold), axis=2)

        # 5. Perform calculations using NumPy
        total_pixels = image_data.shape[0] * image_data.shape[1]
        fire_pixels = np.sum(fire_mask)
        
        if fire_pixels == 0:
            print("No fire detected based on the defined color thresholds.")
            return

        fire_percentage = (fire_pixels / total_pixels) * 100

        print("\n--- Wildfire Analysis Results ---")
        print(f"Total pixels in image: {total_pixels}")
        print(f"Detected fire pixels: {fire_pixels}")
        print(f"Percentage of area covered by fire: {fire_percentage:.2f}%")
        print("---------------------------------")

        # 6. Create a visual representation of the detected fire
        # Create a black image of the same size
        highlighted_fire_image_data = np.zeros_like(image_data)
        
        # Where the fire_mask is True, copy the original pixel color
        highlighted_fire_image_data[fire_mask] = image_data[fire_mask]
        
        # Convert the NumPy array back to a Pillow image
        highlighted_fire_img = Image.fromarray(highlighted_fire_image_data)

        # 7. Display the results using Matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Satellite Image")
        axes[0].axis('off')

        axes[1].imshow(highlighted_fire_img)
        axes[1].set_title("Detected Fire Area")
        axes[1].axis('off')
        
        plt.suptitle("Wildfire Satellite Image Analysis", fontsize=16)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # First, create our sample image data
    # mock_image_file = create_mock_wildfire_image()
    
    # Then, run the analysis on the generated image
    analyze_wildfire_image("sd-3layers.jpg")
